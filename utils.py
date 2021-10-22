import numpy as np
from matplotlib import pyplot as plt


def plot_samples(graph, samples, legend=None, **kwargs):
    """Plot all relevant dependencies in a graph from a/multiple sample(s)."""
    # If we did not already receive a list of samples, make one element list
    if not isinstance(samples, list):
        samples = [samples]
    # Get non root variables
    non_roots = graph.non_roots()
    # Get maximum number of input variables
    max_deps = max([len(graph.parents(var)) for var in non_roots])

    fig, axs = plt.subplots(len(non_roots), max_deps,
                            figsize=(5 * max_deps, 5 * len(non_roots)))

    # Go through all dependencies and plot them as 2D scatter plots
    for i, y_var in enumerate(non_roots):
        for j, x_var in enumerate(graph.parents(y_var)):
            for sample in samples:
                axs[i, j].plot(sample[x_var].numpy(),
                               sample[y_var].numpy(), '.', **kwargs)
                axs[i, j].set_xlabel(x_var)
                axs[i, j].set_ylabel(y_var)
                if legend:
                    axs[i, j].legend(legend)
    plt.tight_layout()
    plt.show()


def combine_variables(variables, sample, as_var=False):
    """Stack variables from sample along new axis."""
    data = torch.cat([sample[i] for i in variables], dim=1)
    if as_var:
        return Variable(data)
    else:
        return data