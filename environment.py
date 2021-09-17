from graph import Graph


class Environment(object):
    def __init__(self, name):
        self.examples = []
        self.name = name
        if self.name == 'causal':
            self.sem = SEM({"Np": None, "Z": None, "Nx": None, "X": ["Np", "Z"], "X": ["Z", "P", "Nx"], "Y": ["P", "X"]})
            self.sem.draw()


    def update(self):
        # T(y,a,s): probability of state distributions when group s currently in state y taking action a 
        pass




class SEM(Graph):
    """
    The class instantiates a graph with equations and distributions 
    """
    def __init__(self, graph):
        super().__init__(graph)
        self.equations = {}
        self.learned = {}

    def sample(self, n_samples):
        sample = {}
        for v in self.topological_sort():
            print("Sample vertex {}...".format(v), end=' ')
            if v in self.roots():
                sample[v] = self.equations[v](n_samples)
            else:
                sample[v] = self.equations[v](sample)
            print("DONE")
        return sample


    def attach_equation(self, vertex, equation):
        """
        Attach an equation or distribution to a vertex.
        In an SEM each vertex is determined by a function of its parents (and
        independent noise), except for root vertices, which follow some
        specified distribution.
        Arguments:
            vertex: The vertex for which we attach the equation.
            equation: A callable with a single argument. For a root vertex,
            this is the number of samples to draw. For non-root vertices the
            argument is a dictionary, where the keys are the parent vertices
            and the values are torch tensors containing the data.
        """
        print("Attaching equation to vertex {}...".format(vertex), end=' ')
        self.equations[vertex] = equation
        print("DONE")


    def intervene(self):
        pass

    
    
