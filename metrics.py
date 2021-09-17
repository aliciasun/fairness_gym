import numpy as np




class State:
    def __init__(self, x, z, y):
        self.x = x
        self.y = y
        self.z = z



class Metrics:
    def __init__(self, state):
        self.state = state

    def base_rate(self):
        return np.mean(self.state.y)

    def fpr(self, pred):
        return np.mean(np.logical_and(pred.round() == 1, self.state.y == 0))

    def fnr(self, pred):
        return np.mean(np.logical_and(pred.round() == 0, self.state.y == 1))

    

