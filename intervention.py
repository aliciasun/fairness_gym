import copy

import torch
from torch.autograd import Variable

from utils import combine_variables

class Interventions:
    # Methods for creating intervened samples
    known_functions = {
        'randn': (lambda self, mean, var:
                  torch.randn(self.n_samples, 1) * var + mean),
        'const': (lambda self, const:
                  torch.ones(self.n_samples, 1) * const),
        'rand': (lambda self, start, end:
                 torch.rand(self.n_samples, 1) * (start - end) + end),
        'range': (lambda self, a, b:
                  torch.linspace(a, b, steps=self.n_samples).unsqueeze_(1)),
        'bernoulli': (lambda self, p:
                      torch.bernoulli(torch.ones(self.n_samples, 1) * p)),
    }

    def __init__(self, sem, samples, intervention_spec, target):
        """
        Example:
        Intervention on vertex T will update the values of T
        Intervention on edge T->X will update the attached equations of X
        spec = {
            'T':{
                'const':[1,0],
                'randn': [(0,3)]
            },
            'TX':{
                new_fun
            }
        }
        """
        self.sem = sem
        self.base_sample = samples
        self.n_samples = len(next(iter(self.base_sample.values())))
        self.interventions = intervention_spec
        self.proxies = list(intervention_spec.keys())
        self.intervened_graph = self.sem.get_intervened_graph(self.proxies)
        self.target = target
        verbose = False
        self.verbose = verbose
        self._set_n_interventions()
        self.training_samples = []
        self._check_input()


    def _check_input(self):
        """Some basic checks of the input."""
        assert self.target in self.sem.leafs(), \
            "Can't correct for non-leaf {}".format(self.target)

        assert self.base_sample[self.target].size(-1) == 1, \
            "Can't correct for multidimensional target {}".format(self.target)

        for proxy in self.proxies:
            if len(proxy) == 1:
                assert self.target in self.sem.descendants(proxy), \
                    ("Can't correct for non-descendant {} of {}."
                    .format(self.target, proxy))


    def _set_n_interventions(self):
        """Set the total number of interventions, i.e. training sets."""
        self.n_interventions = 1
        for proxy, funcs in self.interventions.items():
            n_proxy = 0
            if len(proxy) == 1:
                for params in funcs.values():
                    if not isinstance(params, list):
                        params = [params]
                    n_proxy += len(params)
                self.n_interventions *= n_proxy


    def _create_intervened_samples(self):
        """For each intervention get a sample with the right proxy values."""
        self.training_samples = []
        if self.verbose > 0:
            print("Initialize training samples with intervened values...",
                  end=' ')
        for proxy, functions in self.interventions.items():
            if self.verbose > 1:
                print("Set proxy {} with functions {}"
                      .format(proxy, functions))
            if len(proxy) == 1:
                # variable intervention
                for func, parameters in functions.items():
                    if not isinstance(parameters, list):
                        parameters = [parameters]
                    for params in parameters:
                        sample = copy.deepcopy(self.base_sample)
                        sample[proxy] = self.known_functions[func](self, *params)
                        self.training_samples.append(sample)
            else:
                # policy intervention
                for func in functions:
                    self.sem.attach_equation(proxy[1], func)
                    print("finish policy intervention")
                    print(self.summary())
                    sample = copy.deepcopy(self.base_sample)
                    # update sem equations
                    sample[proxy] = self.sem.equations[proxy[1]](sample)
                    self.training_samples.append(sample)
        print("DONE")

    def _update(self):
        """Update the variables downstream of the proxies."""
        exclude = self.intervened_graph.roots() + [self.target]
        update = [v for v in self.intervened_graph.topological_sort()
                  if v not in exclude]
        if self.verbose > 0:
            print("Predict non-roots {} in all intervened samples."
                  .format(update))
        for sample in self.training_samples:
            self.sem.predict_from_sample(sample, update=update, mutate=True)
        print("All intervened samples updated.")


    def _set_training_samples(self):
        """Generate the training samples for the given interventions."""
        self._create_intervened_samples()
        self._update()


    def _get_parameter_indices(self, vertex):
        """Find the indices of the vertex in the input tensor for target."""
        if self.verbose > 1:
            print("Find parameters indices of {} in all inputs {}"
                  .format(vertex, self.intervened_graph.parents(self.target)))
        parents = self.intervened_graph.parents(self.target)
        start = 0
        for p in parents:
            if p == vertex:
                end = start + self.base_sample[p].size(-1)
                break
            start += self.base_sample[p].size(-1)
        if self.verbose > 1:
            print("Indices from {} to {}".format(start, end))
        return start, end

    def _copy_and_freeze(self, model, biases):
        """Copy a learned model and partially freeze parameters."""
        # Copy the original model
        corrected = copy.deepcopy(model)

        if self.verbose > 1:
            print("Freeze all parameters...", end=' ')
        # First freeze all parameters
        for param in corrected.parameters():
            param.requires_grad = False
        if self.verbose > 1:
            print("DONE")

        if self.verbose > 1:
            if biases:
                print("Retrain weights and biases for {} to {}"
                      .format(self.proxies, self.target))
            else:
                print("Retrain only weights for {} to {}"
                      .format(self.proxies, self.target))

        # Only give gradients to the part that is retrained for correction
        for v in self.intervened_graph.parents(self.target):
            if v in self.proxies:
                if self.verbose > 2:
                    print("Partially retrain parameters of"
                          .format(corrected.layers[0]))
                start, end = self._get_parameter_indices(v)
                for i in range(start, end):
                    if biases:
                        for param in corrected.layers[0][i].parameters():
                            param.requires_grad = True
                    else:
                        corrected.layers[0][i].weight.requires_grad = True
        return corrected

    def train_corrected(self, batchsize=32, epochs=50, biases=False, **kwargs):
        """
        Retrain part of the SEM to correct for undesired discrimination.
        Depending on the specified interventions, generate the intervened
        samples and retrain the functions of the necessary paths in the causal
        graph to remove the overall influence from the proxies on the target.
        Arguments:
            batchsize: The minibatch size for training (default: 32).
            epochs: The number of epochs for training (default: 50).
            biases: A boolean indicating whether to also retrain biases (or
            only the weights) (default: False).
            **kwargs: Further named parameters passed on to `torch.optim.Adam`.
        Returns:
            The corrected network for prediction of the target vertex.
        """
        # Some basic input checks
        target = self.target
        proxies = self.proxies
        parents = self.intervened_graph.parents(target)
        print("Correct for the effect of {} on {}.".format(proxies, target))

        print("Generate intervened samples.")
        self._set_training_samples()
        print("All intervened samples ready for training.")

        # Sanity check
        assert len(self.training_samples) == self.n_interventions, \
            ("# interventions {} does not match # training samples {}"
             .format(self.n_interventions, self.training_samples))
        print("There are {} interventions.".format(len(self.training_samples)))

        print("Freeze everything except first weights from {} to {}..."
              .format(proxies, target), end=' ')
        corrected = self._copy_and_freeze(self.sem.learned[target], biases)
        print("DONE")

        print("Set up the optimizer...", end=' ')
        params = filter(lambda p: p.requires_grad, corrected.parameters())
        opt = torch.optim.Adam(params, **kwargs)
        print("DONE")

        print("Partially retrain the target model for correction...", end=' ')
        n_samples = self.n_samples
        for epoch in tqdm(range(epochs), desc='epochs'):
            p = torch.randperm(n_samples).long()
            # Go through batches
            for i1 in tqdm(range(0, n_samples, batchsize), desc='minibatches',
                           leave=False):
                i2 = min(i1 + batchsize, n_samples)
                # Reset gradients
                opt.zero_grad()
                # Forward pass
                Ys = Variable(torch.zeros(i2 - i1, self.n_interventions))
                for i, sample in enumerate(self.training_samples):
                    data = combine_variables(parents, sample)[p]
                    args = Variable(data[i1:i2, :])
                    Ys[:, i] = corrected(args).squeeze()
                # Compute loss
                loss = torch.sum(torch.var(Ys, dim=1))
                # Backward pass
                loss.backward()
                # Parameter update
                opt.step()
        print("DONE")
        print("Finished correction.")
        return corrected


    def summary(self):
        """Print a summary of the interventions."""
        print("Sample size: {}, Number of interventions {}"
              .format(self.n_samples, self.n_interventions))
        print(self.interventions)
        print("Intervention on structural equations")
        import inspect
        for vertex, equation in self.sem.equations.items():
            print("{}:{}".format(vertex,inspect.getsource(equation)))
    