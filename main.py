import neural_network
import load_data

if __name__ == '__main__':
    laod = load_data.load_data()
    data = laod.load_train_data('./data/train.csv')
    test_data = data[: int(len(data)/0.1)]
    training_data = data[int(len(data)/0.1):]
    net = neural_network.neural_network([784, 30, 40, 10])
    net.sgd(train_data=training_data, epoch=20, mini_batch_size=30, alpha=2.0, test_data= test_data)
