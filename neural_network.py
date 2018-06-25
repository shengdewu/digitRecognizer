import numpy as np

class neural_network(object):
    def __init__(self, size=()):
        if not size:
            raise Exception("Invalid neuro network ")
            return

        self.__weight__ = [np.random.randn(n2, n1) for n1, n2 in zip(size[0:-1], size[1:])]
        self.__bias__ = [np.random.randn(n, 1) for n in size[1:]]
        self.__layers__ = len(size)

    def sgd(self, train_data, test_data, epoch = 10, mini_batch_size = 30, alpha = 2):
        '''
        :param train_data: a list of tuple
        :param epoch: The number of iterations
        :param mini_batch_size: The size of the small batch data
        :param alpha:Learning rate or step length
        :return:
        '''
        iter_cnt = 0
        while iter_cnt < epoch:
            np.random.shuffle(train_data)
            mini_batches = [train_data[n: n+mini_batch_size] for n in range(0, len(train_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.__update_mini_batch__(mini_batch, alpha)

            self.evaluate(test_data, iter_cnt)
            iter_cnt += 1

    def evaluate(self, test_data, iter_cnt):
        '''
        :param test_data: a list of tuple
        :param iter_cnt: iter cnt
        :return:
        '''
        len_data = len(test_data)
        test_reuslt = [(self.__feed_forward__(x), y) for x, y in test_data]
        correct_cnt = sum(int(x==y) for x, y in test_reuslt)
        print('%d %d %d %f' % (iter_cnt, correct_cnt, len_data, float(correct_cnt)/len_data))

        return


    def predict(self, predict_data):
        '''
        :param predict_data: a list of list or tuple
        :return: label of predict_data
        '''
        label = []
        for x in predict_data:
            label.append(self.__feed_forward__(x))
        return label

    def __feed_forward__(self, x):
        active = x
        for w, b in zip(self.__weight__, self.__bias__):
            z = np.dot(w, active) + b
            active = self.__sigmod__(z)
        return np.argmax(active)

    def __update_mini_batch__(self, mini_batch, alpha):
        sum_delta_b = [np.zeros(b.shape) for b in self.__bias__]
        sum_delta_w = [np.zeros(w.shape) for w in self.__weight__]
        for x, y in mini_batch:
            delta_w, delta_b = self.__back_propagation__(y, x)
            sum_delta_b = [sb + b for sb, b in zip(sum_delta_b, delta_b)]
            sum_delta_w = [sw + w for sw, w in zip(sum_delta_w, delta_w)]
        self.__weight__ = [w - alpha * sw / len(mini_batch) for w, sw in zip(self.__weight__ ,sum_delta_w)]
        self.__bias__ = [b - alpha * sb / len(mini_batch) for b, sb in zip(self.__bias__ ,sum_delta_b)]

    def __back_propagation__(self, y, x):
        delta_b = [np.zeros(b.shape) for b in self.__bias__]
        delta_w = [np.zeros(w.shape) for w in self.__weight__]
        #feedforward
        actives = [x]
        zs = []
        active = x
        for w, b in zip(self.__weight__, self.__bias__):
            z = np.dot(w, active) + b
            zs.append(z)
            active = self.__sigmod__(z)
            actives.append(active)

        delta = self.__cost_derivative__(y, actives[-1]) * self.__sigmod_derivative__(zs[-1])
        delta_w[-1] = np.dot(delta, actives[-2].transpose())
        delta_b[-1] = delta
        #backforward
        for lay in range(2, self.__layers__):
            delta = np.dot(self.__weight__[-lay + 1].transpose(), delta) * self.__sigmod_derivative__(zs[-lay])
            delta_w[-lay] = np.dot(delta, actives[-lay-1].transpose())
            delta_b[-lay] = delta

        return delta_w, delta_b

    def __cost_derivative__(self, y, o):
        return o - y

    def __sigmod__(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __sigmod_derivative__(self, z):
        return self.__sigmod__(z) * (1 - self.__sigmod__(z))


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