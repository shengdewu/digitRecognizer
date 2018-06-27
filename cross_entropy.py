import numpy as np
import neurons_network_base

class cross_entropy(neurons_network_base.neurons_network_base):

    def __init__(self, size):
        '''
        :param size: list, the number of neurons in each layer
        '''
        super(cross_entropy, self).__init__(size)
        return

    def back_pro(self, x, y):
        delt_bias = [np.zeros(b.shape) for b in self.bias]
        delt_wight = [np.zeros(w.shape) for w in self.wight]

        #feedforward
        active = x
        actives = [x]
        zs = []
        for w, b in zip(self.wight, self.bias):
            z = np.dot(w, active) + b
            zs.append(z)
            active = self.sigmod(z)
            actives.append(active)

        delt = self.except_error(actives[-1], y)
        delt_wight[-1] = np.dot(delt, actives[-2].transpose())
        delt_bias[-1] = delt

        #backforward
        for lay in range(2, self.lay_num):
            delt = np.dot(self.wight[-lay + 1].transpose(), delt) * self.sigmod_derivative(zs[-lay])
            delt_wight[-lay] = np.dot(delt, actives[-lay-1].transpose())
            delt_bias[-lay] = delt

        return delt_wight, delt_bias
