# TODO

Please note there's a major overhaul of this package taking place:

- [x] Get rid of duality individual-model (simplify classes)
- [x] Rewrite XGBoost
- [x] Rewrite Genetic CNN model
- [ ] Add a scikit-learn model
- [ ] Add module section with predefined parameters
- [ ] Use redis instead of rabbitmq for distributed algorithm
- [x] Adapt to python3.10+ (f-strings, linting, type hinting)
- [x] Simplify dataset retrieval for mnist, iris, fashion-mnist, and others
- [ ] Fix second generation bug in client
- [ ] Add public/private key validation for clients
- [ ] Implement Genetic CNN or other in PyTorch
- [ ] Automate CICD workflows (linting, testing, and publishing) with GitHub Actions
- [ ] Use proper logging instead of prints
- [ ] Create CONTRIBUTE.md
- [ ] Add repo badges

# gentun: genetic algorithm for hyperparameter tuning

The goal of this project is to create a simple framework
for [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) tuning of machine learning models,
like Neural Networks and Gradient Boosting Trees, using a genetic algorithm. Evaluating the fitness of an individual in
a population involves training a model with a specific set of hyperparameters, which is a time-consuming process. To
address this, we employ a client-server approach. Multiple clients can handle model training and cross-validation of
individuals provided by the server. The server manages the generation of offspring through reproduction and mutation.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ [XGBoost tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) on Parameter Tuning

*"[...] The number of possible network structures increases exponentially with the number of layers in the network,
which inspires us to adopt the genetic algorithm to efficiently traverse this large search space."* ~
[Genetic CNN](https://arxiv.org/abs/1703.01513) paper

## :construction: Supported models

This project supports hyperparameter tuning for the following models:

- [x] XGBoost regressor and classifier
- [ ] Scikit-learn regressor and classifier
- [x] [Genetic CNN](https://arxiv.org/pdf/1703.01513.pdf) with Tensorflow
- [ ] [A Genetic Programming Approach to Designing Convolutional Neural Network Architectures](https://arxiv.org/pdf/1704.00764.pdf)

## :construction: Contributing

Feel free to submit your custom `gentun.models.Model` to enhance the project.
You can also help us speed up hyperparameter search with your spare GPU time.
Check [./CONTRIBUTE.md]

You can use as an example the
*XgboostIndividual* and *XgboostModel* classes provided which have a simple gene encoding for instructional purposes.

## Installation

```bash
pip install gentun
```

## Usage

### On a single node

The most basic way to run the algorithm is in a single machine, as shown in the following example where we use it to
find the optimal hyperparameters of an `xgboost` model. First, we download a sample dataset:

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
    RandomChoice("max_depth", range(3, 11)),
    RandomChoice("min_child_weight", range(11)),
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

Finally, we are ready to run our search:

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
be [implemented on the MNIST handwritten digits set](examples/geneticcnn_mnist.py).

Note that in genetic algorithms, the *fitness* of an individual is supposed to be maximized. By default, this framework
follows this convention. Nonetheless, to make the framework more flexible, you can use the `maximize=False` parameter in
`algorithm.run()` to override this behavior and minimize your fitness metric (e.g. when you want to minimize the loss,
for example *rmse* or *binary crossentropy*).

#### :construction: Custom individuals

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
    additional_parameters={'kfold': 3}, maximize=False
)
pop.add_individual(XgboostIndividual(x_train, y_train, genes=custom_genes, kfold=3))
```

#### :construction: Grid search

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
    additional_parameters={'kfold': 3},
    maximize=False
)
```

Running the genetic algorithm on this population for only one generation is equivalent to doing a grid search. Note that
only *XgboostIndividual* is compatible with the *GridPopulation* class.

### Multiple computers - distributed algorithm

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
