# gentun: genetic algorithm for hyperparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models such
as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Since measuring the fitness of a particular
individual of a given population is time consuming, a client-server approach is used to allow several clients perform
model fitting and cross-validation of the individuals that a server assigns. Offspring generation by reproduction and
mutation is handled by the server.

*"Parameter tuning is a dark art in machine learning, the optimal parameters of a model can depend on many scenarios."*
~ XGBoost's Notes on Parameter Tuning

# Supported models (work in progress)

- [ ] XGBoost regressor
- [ ] XGBoost classifier
- [ ] Scikit-learn Multilayer Perceptron Regressor
- [ ] Scikit-learn Multilayer Perceptron Classifier
- [ ] Keras

# Sample usage

Client-server model is still a work in progress, but you can test the genetic algorithm on a single box, as shown in the
following example:

```python
import pandas as pd
from gentun import Population, GeneticAlgorithm
```

```python
# Load features and response variable from train set
data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
y_train = data['quality']
x_train = data.drop(['quality'], axis=1)
```

```python
# Generate a random population and run the genetic algorithm
pop = Population('XgboostIndividual', x_train, y_train, size=100, additional_parameters={'nfold': 3})
ga = GeneticAlgorithm(pop)
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
