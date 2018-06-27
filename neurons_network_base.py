import numpy as np

class neurons_network_base(object):

    def __init__(self, size=None):
        if not size:
            raise Exception("Invalid neuro network ")
            return

        self.lay_num = len(size)
        self.bias = [np.random.randn(x, 1) for x in size[1:]]
        self.wight = [np.random.randn(x, y) for x, y in zip(size[1:], size[:-1])]
        return

    def train(self, train_data=None, epoch=30, nrate=1.5, batch_size=30, test_data=None):
        '''
        :param train_data: a list of tuple, label(y) must be a matrix, lable(y) and x is must be column  vector
        :param epoch: number of iterations
        :param nrate: Learning rate or step length
        :param batch_size: size of the small batch data
        :param test_data: a list of tuple, label(y) must be a real number
        :return:
        '''
        if not train_data:
            raise Exception('Invalid traind data')
            return

        iter_cnt = -1
        while iter_cnt < epoch:
            iter_cnt += 1
            np.random.shuffle(train_data)
            min_batches = [train_data[x:x+batch_size] for x in range(0, len(train_data), batch_size)]
            for min_batch in min_batches:
                self.update_min_batch(min_batch, nrate)

            self.evaluate(test_data, iter_cnt)

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

        delt = self.except_error(actives[-1], y) * self.sigmod_derivative(zs[-1])
        delt_wight[-1] = np.dot(delt, actives[-2].transpose())
        delt_bias[-1] = delt

        #backforward
        for lay in range(2, self.lay_num):
            delt = np.dot(self.wight[-lay + 1].transpose(), delt) * self.sigmod_derivative(zs[-lay])
            delt_wight[-lay] = np.dot(delt, actives[-lay-1].transpose())
            delt_bias[-lay] = delt

        return delt_wight, delt_bias

    def update_min_batch(self, batch, nrate):
        delt_bias_sum = [np.zeros(x.shape) for x in self.bias]
        delt_wight_sum = [np.zeros(x.shape) for x in self.wight]
        for x, y in batch:
            delt_wight, delt_bias = self.back_pro(x, y)
            delt_bias_sum = [sb + b for sb, b in zip(delt_bias_sum, delt_bias)]
            delt_wight_sum = [sw + w for sw, w in zip(delt_wight_sum, delt_wight)]

        self.wight = [w - nrate*dw/len(batch) for w, dw in zip(self.wight, delt_wight_sum)]
        self.bias = [b - nrate*db/len(batch)  for b, db in zip(self.bias, delt_bias_sum)]

        return

    def evaluate(self, test_data, iter_cnt):
        test_result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        correct_number = sum(int(x == y) for x, y in test_result)
        len_data = len(test_data)
        print('%d %d %d %f' % (iter_cnt, correct_number, len_data, float(correct_number) / len_data))
        return

    def predicate(self, data):
        '''
        :param data: a list of list or tuple
        :return: label of predict_data
        '''
        label = []
        for x in data:
            label.append(np.argmax(self.feedforward(x)))
        return label

    def feedforward(self, x):
        active = x
        for w, b in zip(self.wight, self.bias):
            z = np.dot(w, active) + b
            active = self.sigmod(z)
        return active

    def sigmod(self, z):
        return (1.0 /(1.0 + np.exp(-z)))

    def sigmod_derivative(self, z):
        return self.sigmod(z) * (1 - self.sigmod(z))

    def except_error(self, active, y):
        return active - y

