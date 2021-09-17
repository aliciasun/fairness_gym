class Intervention():
    def __init___(self, sem, samples, intervention_spec, target):
        """
        Example:
        Intervention on vertex T and edge T->X
        spec = {
            'T':{
                'const':[1,0],
                'randn': [(0,3)]
            },
            'TX':{
            }
        }
        """
        self.sem = sem
        self.samples = samples
        self.intervention_spec = intervention_spec
        self.proxies = list(intervention_spec.keys())
        self.intervened_graph = self.sem.get_intervened_graph(self.proxies)
        self.target = target
