import os
import csv

class Files(object):

    def loadTrainData(self, path):
        fdata = csv.reader(open(path, 'r'))
        trainData =[]
        trainLabel = []
        for line in fdata:
            if 'label' == line[0]:
                continue
            label = int(line[0])
            preData = line[1:]
            data = [int(i) for i in preData]
            trainData.append(data)
            trainLabel.append(label)
        return trainData, trainLabel

    def loadTestData(self, path):
        fdata = csv.reader(open(path, 'r'))
        testData =[]
        for line in fdata:
            if 'pixel0' == line[0]:
                continue
            data = [int(i) for i in line]
            testData.append(data)
        return testData


