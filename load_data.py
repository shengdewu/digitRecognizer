import os
import csv

class load_data(object):

    def load_train_data(self, path):
        fdata = csv.reader(open(path, 'r'))
        train_x =[]
        train_y = []
        for line in fdata:
            if 'label' == line[0]:
                continue
            data_y = int(line[0])
            data_x = line[1:]
            data_x = [int(i) for i in data_x]
            train_x.append(data_x)
            train_y.append(data_y)
            train_data = [(x, y) for x, y in zip(train_x, train_y)]
        return train_data

    def laod_test_data(self, path):
        fdata = csv.reader(open(path, 'r'))
        test_data =[]
        for line in fdata:
            if 'pixel0' == line[0]:
                continue
            data = [int(i) for i in line]
            test_data.append(data)
        return test_data


