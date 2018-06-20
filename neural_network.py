import numpy as np

class neural_network(object):
    def __init__(self, size=()):
        if not size:
            raise Exception("Invalid neuro network ")
            return

        self.__weight__ = [np.random.randn(n2, n1) for n1, n2 in zip(size[0:2], size[1:])]
        self.__bias__ = [np.random.randn(n, 1) for n in size[1:]]
        self.__layers__ = len(size)

    def backpropagation(self, y, x):
        delta_b = [np.zeros(b.shape) for b in self.__bias__]
        delta_w = [np.zeros(w.shape) for w in self.__weight__]
        #feedforward
        actives = [x]
        zs = []
        active = x
        for w, b in zip(self.__weight__, self.__bias__):
            z = np.dot(w, active) + b
            zs.append(z)
            active = self.sigmod(z)
            actives.append(active)

        delta = self.cost_derivative(y, actives[-1]) * self.sigmod_derivative(zs[-1])
        delta_w[-1] = np.dot(delta, actives[-2].transpose())
        delta_b[-1] = delta
        #backforward
        for lay in range(2, self.__layers__):
            delta = np.dot(self.__weight__[-lay + 1].transpose(), delta) * self.sigmod_derivative(zs[-lay])
            delta_w[-lay] = np.dot(delta, actives[-lay-1].transpose())
            delta_b[-lay] = delta

        return delta_w, delta_b

    def cost_derivative(self, y, o):
        return y - o

    def sigmod(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmod_derivative(self, z):
        return self.sigmod(z) * (1 - self.sigmod(z))


def test():
    try:
        nn = neural_network([20, 30, 10])
        x = np.floor(np.random.randn(100, 20) % 10)
        y = np.floor(np.random.randn(100, 10) % 10)

        for x1, y1 in zip(x, y):
            x1 = x1.reshape(x1.shape[0], 1)
            y1 = y1.reshape(y1.shape[0], 1)
            nn.backpropagation(y1, x1)


    except Exception as err:
        print(err)