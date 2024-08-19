# gentun: distributed genetic algorithm for hyperparameter tuning

The goal of this project is to create a simple framework
for [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) tuning of machine learning models,
like Neural Networks and Gradient Boosting Trees, using a genetic algorithm. Evaluating the fitness of an individual in
a population involves training a model with a specific set of hyperparameters, which is a time-consuming process. To
address this, we provide a client-server approach. Multiple clients can handle model training and cross-validation of
individuals provided by a server while the server manages the generation of offspring through reproduction and
mutation.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ [XGBoost tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) on Parameter Tuning

*"[...] The number of possible network structures increases exponentially with the number of layers in the network,
which inspires us to adopt the genetic algorithm to efficiently traverse this large search space."* ~
[Genetic CNN](https://arxiv.org/abs/1703.01513) paper

## :construction: Supported models

This project supports hyperparameter tuning for the following models:

- [x] XGBoost regressor and classifier
- [x] Scikit-learn regressor and classifier
- [x] [Genetic CNN](https://arxiv.org/pdf/1703.01513.pdf) with Tensorflow
- [ ] [A Genetic Programming Approach to Designing Convolutional Neural Network Architectures](https://arxiv.org/pdf/1704.00764.pdf)

## :construction: Contributing

Feel free to submit your custom [`gentun.models.Model`](src/gentun/models/base.py#L9-L25)
and [`gentun.genes.Gene`](src/gentun/genes.py#L12-L44) subclasses to enhance the project. You can also help us speed up
hyperparameter search with your spare GPU time. Check our documentation on [how to contribute](./CONTRIBUTE.md).

## :construction: Installation

```bash
pip install gentun
```

## Usage

### On a single node

The most basic way to run the algorithm is using a single machine, as shown in the following example where we use it to
find the optimal hyperparameters of an [`xgboost`](https://xgboost.readthedocs.io/en/stable/) model. First, we download
a sample dataset:

```python
from sklearn.datasets import load_iris

data = load_iris()
x_train = data.data
y_train = data.target
```

Next, we need to define the hyperparameters we want to optimize:

```python
from gentun.genes import RandomChoice, RandomLogUniform

genes = [
    RandomLogUniform("learning_rate", minimum=0.001, maximum=0.1, base=10),
    RandomChoice("max_depth", [3, 4, 5, 6, 7, 8, 9, 10]),
    RandomChoice("min_child_weight", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
]
```

We are using the `gentun.models.xgboost.XGBoostCV` model, which performs k-fold cross validation with available train
data and returns an average metric over the folds. Thus, we need to define some static parameters which are shared
across the population over all generations:

```python
kwargs = {
    "booster": "gbtree",
    "objective": "multi:softmax",
    "metrics": "mlogloss",  # The metric we want to minimize with the algorithm
    "num_class": 3,
    "nfold": 5,
    "num_boost_round": 5000,
    "early_stopping_rounds": 100,
}
```

Finally, we are ready to run our genetic algorithm. `gentun` will check that all the model's required parameters are
passed either through genes or keyword arguments.

```python
from gentun.algorithms import Tournament
from gentun.models.xgboost import XGBoostCV
from gentun.populations import Population

# Run the genetic algorithm with a population of 50 for 100 generations
population = Population(genes, XGBoostCV, x_train, y_train, 50, **kwargs)
algorithm = Tournament(population)
algorithm.run(100, maximize=False)
```

As shown above, when the model and genes are implemented, experimenting with the genetic algorithm is simple. See for
example how easily can the Genetic CNN paper
be [defined on the MNIST handwritten digits set](examples/geneticcnn_mnist.py).

Note that in genetic algorithms, the *fitness* of an individual is a number to be maximized. By default, this framework
follows this convention. Nonetheless, to make the framework more flexible, you can use the `maximize=False` parameter in
`algorithm.run()` to override this behavior and minimize your fitness metric (e.g. when you want to minimize the loss,
for example *rmse* or *binary crossentropy*).

#### Adding pre-defined individuals

Oftentimes, it's convenient to initialize the genetic algorithm with some known individuals instead of a random
population. You can add custom individuals to the population before running the genetic algorithm if you already have
an intuition of which hyperparameters work well with your model:

```python
from gentun.models.xgboost import XGBoostCV
from gentun.populations import Population


# Best known parameters
hyperparams = {
    "learning_rate": 0.1,
    "max_depth": 9,
    "min_child_weight": 1,
}

# Generate a random population and then add a custom individual
population = Population(genes, XGBoostCV, x_train, y_train, 49, **kwargs)
population.add_individual(hyperparams)
```

#### Performing a grid search

Grid search is also widely used for hyperparameter optimization. This framework provides `gentun.populations.Grid`,
which can be used to conduct a grid search over a single generation pass. You must use genes which define the `sample()`
method, so that uniformly distributed hyperparameter values are obtained with it.

```python
from gentun.genes import RandomChoice, RandomLogUniform
from gentun.models.xgboost import XGBoostCV
from gentun.populations import Grid


genes = [
    RandomLogUniform("learning_rate", minimum=0.001, maximum=0.1, base=10),
    RandomChoice("max_depth", [3, 4, 5, 6, 7, 8, 9, 10]),
    RandomChoice("min_child_weight", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
]

gene_samples = [10, 8, 11]  # How many samples we want to get from each gene

# Generate a grid of individuals
population = Grid(genes, XGBoostCV, x_train, y_train, gene_samples, **kwargs)
```

Running the genetic algorithm on this population for just one generation is equivalent to doing a grid search over 10
`learning_rate` values, all `max_depth` values between 3 and 10, and all `min_child_weight` values between 0 and 10.

### :construction: Distributed algorithm - using multiple nodes

You can speed up the genetic algorithm by using several machines to evaluate models. One of them will act as a *server*,
generating a population and running the genetic algorithm. Each time this *server* needs to evaluate an individual, it
will send a request to a pool of *clients*, which receive the model's hyperparameters and perform model fitting using
k-fold cross-validation. The more *clients* you use, the faster the algorithm will run.

#### Redis setup

```shell
docker run -d --name gentun-redis -p 6379:6379 redis
```

#### Running the distributed genetic algorithm

To run the distributed genetic algorithm, define either a *DistributedPopulation* or a *DistributedGridPopulation* which
will serve as the *server* node. It will send job requests to the message broker each time a set of individuals needs to
be evaluated and will wait until all jobs are completed to produce the next generation of individuals.

```python
from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual

population = DistributedPopulation(
    XgboostIndividual, size=100, additional_parameters={'kfold': 3}, maximize=False,
    host='<rabbitmq_server_ip>', user='<server_username>', password='<server_password>',
    rabbit_queue='<rabbit_queue>'
)
# Run the algorithm for ten generations using client nodes to evaluate individuals
ga = GeneticAlgorithm(population)
ga.run(10)
```

The client nodes are defined using the *GentunClient* class and passing the corresponding individual to it. Each node
has to have access to the train data. You can use as many nodes as desired as long as they have network access to the
message broker server.

```python
from sklearn.datasets import fetch_california_housing
from gentun import GentunClient, XgboostIndividual

data = fetch_california_housing()
y_train = data.target
x_train = data.data

gc = GentunClient(
    XgboostIndividual, x_train, y_train, host='<rabbitmq_server_ip>',
    user='<client_username>', password='<client_password>',
    rabbit_queue='<rabbit_queue>'
)
gc.work()
```

## References

### Genetic algorithms

* Artificial Intelligence: A Modern Approach. 3rd edition. Section 4.1.4
* https://github.com/DEAP/deap
* http://www.theprojectspot.com/tutorial-post/creating-a-genetic-algorithm-for-beginners/3

### XGBoost parameter tuning

* http://xgboost.readthedocs.io/en/latest/parameter.html
* http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

### Papers

* Lingxi Xie and Alan L. Yuille, [Genetic CNN](https://arxiv.org/abs/1703.01513)
* Masanori Suganuma, Shinichi Shirakawa, and Tomoharu
  Nagao, [A Genetic Programming Approach to Designing Convolutional Neural Network Architectures](https://arxiv.org/abs/1704.00764)
