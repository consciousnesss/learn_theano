
import theano.tensor as T
from blocks.bricks import Linear, Rectifier, Softmax, MLP
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


def run_bricks_mnist_mlp(n_epochs=10):
    x = T.matrix('features')
    y = T.lmatrix('targets')

    input_to_hidden = Linear(name='input_to_hidden', input_dim=784, output_dim=100)
    h = Rectifier().apply(input_to_hidden.apply(x))
    hidden_to_output = Linear(name='hidden_to_output', input_dim=100, output_dim=10)
    y_hat = Softmax().apply(hidden_to_output.apply(h))

    cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

    model = ComputationGraph(cost)

    # The following is quite verbose.. Probably there is a function for that
    # this filters all variables by role
    W1, W2 = VariableFilter(roles=[WEIGHT])(model.variables)
    cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
    cost.name = 'cost_with_regularization'

    # initialize weights.
    # If you happened to forget this, you just quitely get nans as output.
    # If you forgot to initialize only a single layer, you still get nan quietly:)
    input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
    input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
    input_to_hidden.initialize()
    hidden_to_output.initialize()

    # the following is the shortcut for MLP
    mlp = MLP(activations=[Rectifier(), Softmax()], dims=[784, 100, 10]).apply(x)
    # but it is not clear how to do weight initializations..

    mnist = MNIST(("train",))
    data_stream = Flatten(DataStream.default_stream(
        mnist, iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))

    algorithm = GradientDescent(cost=cost, parameters=model.parameters,
                                step_rule=Scale(learning_rate=0.1))

    mnist_test = MNIST(("test",))
    data_stream_test = Flatten(DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(
        mnist_test.num_examples, batch_size=1024)))

    monitor = DataStreamMonitoring(
        variables=[cost], data_stream=data_stream_test, prefix="test")

    main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                         extensions=[monitor, FinishAfter(after_n_epochs=n_epochs), Printing()])

    main_loop.run()


if __name__ == '__main__':
    run_bricks_mnist_mlp(n_epochs=1)
    '''
    Impression:
     - very verbose API. Its probably as verbose as to write theano directly.
     - very clumsy API. It feels like you write java - there are 17 imports to run mlp on mnist,
        there are no containers etc.
     - explicit model lifecycle (create, configure, initialize etc) which you need to manually perform
     - parameter initialization is really really crappy. You forget to init one layer - you get silent nans
      as output
     - no shape inference
     - uses Fuel, fuel is verbose and requires unnecessary configuration. Is it so hard to just create a folder
      in user's home and download data there? Why would user need to download and then convert data?
     - all client logic (like MLP wrapper) is in __init__.py files. This is just really really strange..
     - has built in bricks for attention:
      https://blocks.readthedocs.org/en/latest/api/bricks.html#module-blocks.bricks.attention
    '''
