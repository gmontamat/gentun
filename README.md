# gentun: genetic algorithm for hyperparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models such
as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Measuring the fitness of an individual of a
given population implies training the machine learning model using a particular set of parameters which define the
individual's genes. This is a time consuming process, therefore, a master-workers approach is used to allow several
clients (workers) perform the model fitting and cross-validation of individuals passed by a server (master). Offspring
generation by reproduction and mutation is handled by the server.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ XGBoost's Notes on Parameter Tuning

# Supported models (work in progress)

- [x] XGBoost regressor
- [ ] XGBoost classifier
- [ ] Scikit-learn Multilayer Perceptron Regressor
- [ ] Scikit-learn Multilayer Perceptron Classifier
- [ ] Keras

# Sample usage

## Single machine

You can run the genetic algorithm on a single box, as shown in
the following example:

```python
import pandas as pd
from gentun import GeneticAlgorithm, Population, XgboostIndividual
```

```python
# Load features and response variable from train set
data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
y_train = data['quality']
x_train = data.drop(['quality'], axis=1)
```

```python
# Generate a random population and run the genetic algorithm
pop = Population(XgboostIndividual, x_train, y_train, size=100, additional_parameters={'nfold': 3})
ga = GeneticAlgorithm(pop)
ga.run(10)
```

## Multiple boxes

You can speed up the algorithm by using several machines. One of them will act as a *master*, generating a population
and running the genetic algorithm. Each time the *master* needs to evaluate an individual, it will send a request to a
pool of *workers*, which receive the model's hyperparameters from the individual and perform model fitting using n-fold
cross-validation. The more *workers* you use, the faster the algorithm will run.

First, you need to setup a [RabbitMQ](https://www.rabbitmq.com/download.html) message broker server. It will handle
communications between the *master* and all the *workers* via a queueing system.

```bash
$ sudo apt-get install rabbitmq-server
$ sudo service rabbitmq-server start
$ sudo rabbitmqctl add_user <username> <password>
```

Next, start the worker nodes. Each node has to have access to the train data. You can use as many nodes as desired as
long as they can access the RabbitMQ server.

```python
from gentun import GentunWorker, XgboostRegressor
import pandas as pd

data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
y = data['quality']
x = data.drop(['quality'], axis=1)

gw = GentunWorker(XgboostRegressor, x, y)
gw.work()
```

Finally run the genetic algorithm but this time with a *DistributedPopulation* which acts as the *master* node sending
job requests to the *workers* each time an individual needs to be evaluated.

```python
from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual
population = DistributedPopulation(XgboostIndividual, size=100, additional_parameters={'nfold': 3})
ga = GeneticAlgorithm(population)
ga.run(10)
```

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
