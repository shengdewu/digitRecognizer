import neural_network
import load_data
import neurons_network_base
import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import pp

def test_theano():
    #basic
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x,y],z)

    print(f(2,3))
    print(pp(z))

    #matrix
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = function([x, y], z)
    print(f(np.arange(12).reshape((3,4)), 10 * np.ones((3,4))))
    print(pp(z))

    x = T.dscalar('x')
    y = x ** 2
    g = T.grad(y, x)
    print(pp(g))

    k = T.iscalar('k')
    A = T.vector('A')

    result, updates = theano.scan(fn = lambda x, A :  x * A,
                                    outputs_info=T.ones_like(A),
                                    non_sequences=A,
                                    n_steps=k)

    power = theano.function(inputs=[A,k], outputs=result,updates=updates)

    print(power(range(10), 2))

    coefficients = theano.tensor.vector("coefficients")
    x = T.scalar("x")

    max_coefficients_supported = 10000

    print(theano.tensor.arange(max_coefficients_supported))

    # Generate the components of the polynomial
    components, updates = theano.scan(
        fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
        outputs_info=None,
        sequences=[coefficients, theano.tensor.arange(max_coefficients_supported)],
        non_sequences=x)
    # Sum them up
    polynomial = components.sum()

    # Compile a function
    calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

    # Test
    test_coefficients = np.asarray([1, 0, 2], dtype=np.float32)
    test_value = 3
    print(calculate_polynomial(test_coefficients, test_value))
    print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))

    return



if __name__ == '__main__':
    # load = load_data.load_data()
    # training_data, test_data = load.load_train_data('E:/workspace/digitRecognizer/data/train.csv')
    # # net = neural_network.neural_network([784, 30, 40, 10])
    # # net.sgd(train_data=training_data, epoch=20, mini_batch_size=30, alpha=2.0, test_data= test_data)
    # net = neurons_network_base.neurons_network_base([784, 30, 40, 10])
    # net.train(train_data=training_data, epoch=20, batch_size=30, nrate=2.0, test_data= test_data)
    #
    # test_data = load.laod_test_data('E:/workspace/digitRecognizer/data/test.csv')
    # #label = net.predict(test_data)
    # label = net.predicate(test_data)
    #
    # load.save_predict(label=label, path='E:/workspace/digitRecognizer/data/predict_label.csv')

    test_theano()
