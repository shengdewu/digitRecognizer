from src import network
from src import mnist_loader
import neural_network

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #net = network.Network([784, 30, 10])
    #net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    my_net = neural_network.neural_network([784, 30, 40, 10])
    my_net.sgd(train_data=training_data, epoch=20, mini_batch_size=30, alpha=2.0, test_data= test_data)




