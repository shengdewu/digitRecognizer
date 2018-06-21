from load_data import  Files

def digitRecognizer():
    path = 'data/train.csv'
    fdata = Files()
    trainData, trainLabel = fdata.loadTrainData(path)

    path = 'data/test.csv'
    testData = fdata.loadTestData(path)
    return


digitRecognizer()
