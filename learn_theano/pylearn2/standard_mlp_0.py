import pylearn2.train
import pylearn2.training_algorithms.sgd
import pylearn2.termination_criteria
import pylearn2.training_algorithms.learning_rule
import pylearn2.costs.cost
import pylearn2.train_extensions.best_params

from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP, RectifiedLinear, Softmax


def standard_mlp_run(max_epochs=10000):
    model = MLP(
        layers=[
            RectifiedLinear(layer_name='h0', dim=500, sparse_init=15),  # sparse_init - 15 random weights per units are not zeros
            RectifiedLinear(layer_name='h1', dim=500, sparse_init=15),
            Softmax(layer_name='y', n_classes=10, irange=0)  # irange - initalize weights to zero
        ],
        nvis=784
    )

    dataset = MNIST(which_set='train', start=0, stop=50000)

    experiment = pylearn2.train.Train(
        dataset=dataset,
        model=model,
        algorithm=pylearn2.training_algorithms.sgd.SGD(
            batch_size=100,
            learning_rate=0.01,
            monitoring_dataset={
                'train': dataset,
                'valid': MNIST(which_set='train', start=50000, stop=60000),
                'test': MNIST(which_set='test')},
            cost=pylearn2.costs.cost.SumOfCosts(
                costs=[pylearn2.costs.mlp.Default(),
                       pylearn2.costs.mlp.WeightDecay(coeffs={'h0': 0.00005, 'h1': 0.00005, 'y': 0.00005})
                       ]),
            learning_rule=pylearn2.training_algorithms.learning_rule.Momentum(init_momentum=.5),
            termination_criterion=pylearn2.termination_criteria.And(
                criteria=[
                    pylearn2.termination_criteria.MonitorBased(
                        channel_name="valid_y_misclass",
                        prop_decrease=0.,
                        N=10
                    ),
                    pylearn2.termination_criteria.EpochCounter(max_epochs=max_epochs)]),
        ),
        extensions=[
            pylearn2.train_extensions.best_params.MonitorBasedSaveBest(
                channel_name='valid_y_misclass',
                save_path="mlp_3_best.pkl"),
            pylearn2.training_algorithms.learning_rule.MomentumAdjustor(
                start=1,
                saturate=10,
                final_momentum=.99,
            )]
    )
    experiment.main_loop()


if __name__ == '__main__':
    standard_mlp_run()
