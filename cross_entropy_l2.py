import numpy as np
import cross_entropy

class cross_entropy_l2(cross_entropy.cross_entropy):

    def __init__(self, size):
        super(cross_entropy_l2, self).__init__(size)

    def train(self, train_data=None, epoch=30, nrate=1.5, batch_size=30, lamd = 0.1, test_data=None):
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
        nsample = len(train_data)
        iter_cnt = -1
        while iter_cnt < epoch:
            iter_cnt += 1
            np.random.shuffle(train_data)
            min_batches = [train_data[x:x+batch_size] for x in range(0, len(train_data), batch_size)]
            for min_batch in min_batches:
                self.update_min_batch(min_batch, nrate, lamd, nsample)

            self.evaluate(test_data, iter_cnt)

    def update_min_batch(self, batch, nrate, lamd, nsample):
        delt_bias_sum = [np.zeros(x.shape) for x in self.bias]
        delt_wight_sum = [np.zeros(x.shape) for x in self.wight]
        for x, y in batch:
            delt_wight, delt_bias = self.back_pro(x, y)
            delt_bias_sum = [sb + b for sb, b in zip(delt_bias_sum, delt_bias)]
            delt_wight_sum = [sw + w for sw, w in zip(delt_wight_sum, delt_wight)]

        self.wight = [(1-nrate*lamd/nsample)*w - nrate*dw/len(batch) for w, dw in zip(self.wight, delt_wight_sum)]
        self.bias = [b - nrate*db/len(batch) for b, db in zip(self.bias, delt_bias_sum)]

        return