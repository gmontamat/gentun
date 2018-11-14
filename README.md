# gentun: genetic algorithm for hyperparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models such
as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Measuring the fitness of an individual of a
given population implies training the machine learning model using a particular set of parameters which define the
individual's genes. This is a time consuming process, therefore, a server-client approach is used to allow multiple
clients perform the model fitting and cross-validation of individuals passed by a server. Offspring generation by
reproduction and mutation is handled by the server.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ XGBoost's Notes on Parameter Tuning

*"[...] The number of possible network structures increases exponentially with the number of layers in the network,
which inspires us to adopt the genetic algorithm to efficiently traverse this large search space."* ~
[Genetic CNN](https://arxiv.org/abs/1703.01513) paper

# Supported gene encodings

We encourage you to submit your own individual-model pairs to enhance the project. You can base your work on the
*XgboostIndividual* and *XgboostModel* classes provided which have a simple gene encoding for instructional purposes. So
far, this project supports parameter tuning for the following models:

- [x] XGBoost regressor (custom gene encoding)
- [x] XGBoost classifier (custom gene encoding)
- [x] [Genetic CNN](https://arxiv.org/pdf/1703.01513.pdf) using Keras
- [ ] [A Genetic Programming Approach to Designing Convolutional Neural Network Architectures](https://arxiv.org/pdf/1704.00764.pdf)

# Installation

Using a [virtual environment](https://docs.python.org/3.6/tutorial/venv.html) is highly recommended. Also, it is better
to install [xgboost](https://xgboost.readthedocs.io/en/latest/build.html) and
[TensorFlow](https://www.tensorflow.org/install/) before the setup script tries to do it for you because this offers
better customization and also because *pip* may not be able to compile those libraries. Although the module was
originally written for Python 2.7, __only Python 3.6 is currently supported__.

```bash
$ git clone https://github.com/gmontamat/gentun
$ cd gentun
$ python setup.py install
```

# Usage

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
pop = Population(
    XgboostIndividual, x_train, y_train, size=100,
    additional_parameters={'nfold': 3}, maximize=False
)
# Run the algorithm for ten generations
ga = GeneticAlgorithm(pop)
ga.run(10)
```

As seen above, once the individual is defined and its encoding implemented, experimenting with the genetic algorithm is
simple. See for example how easily can the GeneticCNN algorithm be
[implemented on the MNIST handwritten digits set](tests/test_mnist.py).

Note that in Genetic Algorithms, the *fitness* of an individual is supposed to be maximized. By default, this framework
follows the convention. Nonetheless, to make the *Population* class and its variants more flexible, you can set the
parameter `maximize=False` to override this behavior and minimize your fitness metric (so as to minimize the loss, for
example *rmse* or *binary crossentropy*).

### Custom individuals and grid search

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
pop = Population(
    XgboostIndividual, x_train, y_train, size=99,
    additional_parameters={'nfold': 3}, maximize=False
)
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
pop = GridPopulation(
    XgboostIndividual, genes_grid=grid,
    additional_parameters={'nfold': 3},
    maximize=False
)
```

Running the genetic algorithm on this population for only one generation is equivalent to doing a grid search. Note that
only *XgboostIndividual* is compatible with the *GridPopulation* class.

## Multiple computers - distributed algorithm

You can speed up the genetic algorithm by using several machines to evaluate models. One of them will act as a *server*,
generating a population and running the genetic algorithm. Each time this *server* needs to evaluate an individual, it
will send a request to a pool of *clients*, which receive the model's hyperparameters and perform model fitting using
n-fold cross-validation. The more *clients* you use, the faster the algorithm will run.

### Basic RabbitMQ installation and setup

First, you need to install and run [RabbitMQ](https://www.rabbitmq.com/download.html), a message broker server. It will
handle communications between the *server* and all the *client* nodes via a queueing system.

```bash
$ sudo apt-get install rabbitmq-server
$ sudo service rabbitmq-server start
```

Next, you should add a user with write privileges for the *server*. The default guest user can only be used to access
RabbitMQ locally, it is advisable to remove this user.

```bash
$ sudo rabbitmqctl add_user <server_username> <server_password>
$ sudo rabbitmqctl set_permissions -p / <server_username> ".*" ".*" ".*"
```

Also, add a user with fewer privileges to be used by the *client* nodes. You need to name the queue used by the *server*
to send job requests, which is defined by the `rabbit_queue` parameter, whose default value is **rpc_queue**.

```bash
$ sudo rabbitmqctl add_user <client_username> <client_password>
$ sudo rabbitmqctl set_permissions -p / <client_username> "(<rabbit_queue>|amq\.default)" "(<rabbit_queue>|amq\.default)" "(<rabbit_queue>|amq\.default)"
```

Optionally, you can enable an HTTP admin page to configure and monitor RabbitMQ. You can monitor queues and handle user
permissions with a more intuitive web UI.

```bash
$ sudo rabbitmq-plugins enable rabbitmq_management
```

Once enabled, navigate to `<rabbitmq_server_ip>:15672` in your browser to use the web UI. Finally, restart the server to
reflect these changes.

```bash
$ sudo service rabbitmq-server restart
```

### Running the distributed genetic algorithm

To run the distributed genetic algorithm, define either a *DistributedPopulation* or a *DistributedGridPopulation* which
will serve as the *server* node. It will send job requests to the message broker each time a set of individuals needs to
be evaluated and will wait until all jobs are completed to produce the next generation of individuals.

```python
from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual

population = DistributedPopulation(
    XgboostIndividual, size=100, additional_parameters={'nfold': 3}, maximize=False,
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
from gentun import GentunClient, XgboostIndividual
import pandas as pd

data = pd.read_csv('./tests/data/winequality-white.csv', delimiter=';')
y = data['quality']
x = data.drop(['quality'], axis=1)

gw = GentunClient(
    XgboostIndividual, x, y, host='<rabbitmq_server_ip>',
    user='<client_username>', password='<client_password>',
    rabbit_queue='<rabbit_queue>'
)
gw.work()
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

## Papers

* Lingxi Xie and Alan L. Yuille, [Genetic CNN](https://arxiv.org/abs/1703.01513)
* Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, [A Genetic Programming Approach to Designing Convolutional Neural Network Architectures](https://arxiv.org/abs/1704.00764)

## Server-client model and RabbitMQ

* https://www.rabbitmq.com/tutorials/tutorial-six-python.html
