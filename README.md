# gentun: genetic algorithm for hyperparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models such
as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Measuring the fitness of an individual of a
given population implies training the machine learning model using a particular set of parameters which define the
individual's genes. This is a time consuming process, therefore, a master-workers approach is used to allow several
clients (workers) perform the model fitting and cross-validation of individuals passed by a server (master). Offspring
generation by reproduction and mutation is handled by the server.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ XGBoost's Notes on Parameter Tuning

*"[...] The number of possible network structures increases exponentially with the number of layers in the network,
which inspires us to adopt the genetic algorithm to efficiently traverse this large search space."* ~
[Genetic CNN](https://arxiv.org/abs/1703.01513) paper

# Supported gene encodings (work in progress)

We encourage you to submit your own individual-model pairs to enhance the project. You can base your work on the
*XgboostIndividual* and *XgboostModel* classes provided which have a simple gene encoding for instructional purposes. So
far, this project supports parameter tuning for the following models:

- [x] XGBoost regressor (custom gene encoding)
- [x] XGBoost classifier (custom gene encoding)
- [ ] [Genetic CNN](https://arxiv.org/pdf/1703.01513.pdf) using Keras (almost finished)

# Installation

Using a [virtual environment](https://virtualenv.pypa.io) is highly recommended. Also, it is better to install
[xgboost](https://xgboost.readthedocs.io/en/latest/build.html) and [TensorFlow](https://www.tensorflow.org/install/)
before the setup script tries to do it for you because this offers better customization and also because *pip* may not
be able to compile those libraries. Although the module was originally written for Python 2.7, only Python 3 is
currently supported.

```bash
$ git clone https://github.com/gmontamat/gentun
$ cd gentun
$ python setup.py install
```

# Sample usage

## Single machine

The genetic algorithm can be run on a single computer, as shown in the following example:

```python
import pandas as pd
from gentun import GeneticAlgorithm, Population, XgboostIndividual
```

```python
# Load features and response variable from train set
data = pd.read_csv('./tests/data/winequality-white.csv', delimiter=';')
y_train = data['quality']
x_train = data.drop(['quality'], axis=1)
```

```python
# Generate a random population
pop = Population(XgboostIndividual, x_train, y_train, size=100, additional_parameters={'nfold': 3})
# Run the algorithm for ten generations
ga = GeneticAlgorithm(pop)
ga.run(10)
```

Note that in Genetic Algorithms, the *fitness* of an individual is supposed to be maximized. By default in this
framework, the fittest individual of a population is the one with the lowest fitness value (so as to minimize the loss,
for example *rmse* or *binary crossentropy*). To make the *Population* class more flexible, you can pass the
parameter **minimize=False** to override this behaviour and maximize your fitness metric instead.

## Custom individuals and grid search

It's usually convenient to initialize the genetic algorithm with some known individuals instead of a random population.
For example, you can add custom individuals to the population before running the genetic algorithm if you already have
an intuition of which hyperparameters work well with your model:

```python
# Best known parameters so far
custom_genes = {
    'eta': 0.1, 'min_child_weight': 1, 'max_depth': 9,
    'gamma': 0.0, 'max_delta_step': 0, 'subsample': 1.0,
    'colsample_bytree': 0.9, 'colsample_bylevel': 1.0,
    'lambda': 1.0, 'alpha': 0.0, 'scale_pos_weight': 1.0
}
# Generate a random population and add a custom individual
pop = Population(XgboostIndividual, x_train, y_train, size=99, additional_parameters={'nfold': 3})
pop.add_individual(XgboostIndividual(x_train, y_train, genes=custom_genes, nfold=3))
```

Moreover, you can create a grid by defining which values you want to evaluate per gene and the *GridPopulation* class
will generate all possible gene combinations and assign each of them to an individual. This way of generating an initial
population resembles the grid search method which is widely used in parameter optimization:

```python
# Specify which values you want to use, the remaining genes will take the default one
grid = {
    'eta': [0.001, 0.005, 0.01, 0.015, 0.2],
    'max_depth': range(3, 11),
    'colsample_bytree': [0.80, 0.85, 0.90, 0.95, 1.0]
}
# Generate a grid of individuals as the population
pop = GridPopulation(XgboostIndividual, genes_grid=grid, additional_parameters={'nfold': 3})
```

Running the genetic algorithm on this population for only one generation is equivalent to doing a grid search.

## Multiple computers

You can speed up the algorithm by using several machines. One of them will act as a *master*, generating a population
and running the genetic algorithm. Each time the *master* needs to evaluate an individual, it will send a request to a
pool of *workers*, which receive the model's hyperparameters from the individual and perform model fitting using n-fold
cross-validation. The more *workers* you use, the faster the algorithm will run.

First, you need to setup a [RabbitMQ](https://www.rabbitmq.com/download.html) message broker server. It will handle
communications between the *master* and all the *workers* via a queueing system.

```bash
$ sudo apt-get install rabbitmq-server
```

Start the message server and add a user with privileges to communicate the master and worker nodes. The default guest
user can only be used to access RabbitMQ locally, so the first time you start it, you should add a new user and set its
privileges as shown below:

```bash
$ sudo service rabbitmq-server start
$ sudo rabbitmqctl add_user <username> <password>
$ sudo rabbitmqctl set_user_tags <username> administrator
$ sudo rabbitmqctl set_permissions -p / <username> ".*" ".*" ".*"
```

Next, start the worker nodes. Each node has to have access to the train data. You can use as many nodes as desired as
long as they have network access to the message broker server.

```python
from gentun import GentunWorker, XgboostModel
import pandas as pd

data = pd.read_csv('./tests/data/winequality-white.csv', delimiter=';')
y = data['quality']
x = data.drop(['quality'], axis=1)

gw = GentunWorker(
    XgboostModel, x, y, host='<rabbitmq_server_ip>',
    user='<username>', password='<password>'
)
gw.work()
```

Finally, run the genetic algorithm but this time with a *DistributedPopulation* or a *DistributedGridPopulation* which
acts as the *master* node sending job requests to the *workers* each time an individual needs to be evaluated.

```python
from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual

population = DistributedPopulation(
    XgboostIndividual, size=100, additional_parameters={'nfold': 3},
    host='<rabbitmq_server_ip>', user='<username>', password='<password>'
)
# Run the algorithm for ten generations using worker nodes to evaluate individuals
ga = GeneticAlgorithm(population)
ga.run(10)
```

**NOTE:** Future versions will adopt [Apache Kafka](https://kafka.apache.org/) as a message broker in favor of RabbitMQ.

# References

## Genetic algorithms

* Artificial Intelligence: A Modern Approach. 3rd edition. Section 4.1.4
* https://github.com/DEAP/deap
* http://www.theprojectspot.com/tutorial-post/creating-a-genetic-algorithm-for-beginners/3

## XGBoost parameter tuning

* http://xgboost.readthedocs.io/en/latest/parameter.html
* http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

## Master-Workers model and RabbitMQ

* https://www.rabbitmq.com/tutorials/tutorial-six-python.html
