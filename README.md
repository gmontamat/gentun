# gentun: genetic algorithm for hyperparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models
such as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Since measuring the fitness of a
particular individual of a given population is time consuming, a client-server approach is used to allow several
clients perform model fitting and cross-validation of the individuals that a server assigns. Offspring generation by
reproduction and mutation is handled by the server.

# Supported models

* xgboost

# References

## Genetic algorithms

* Artificial Intelligence: A Modern Approach. 3rd edition. Section 4.1.4
* https://github.com/DEAP/deap
* http://www.theprojectspot.com/tutorial-post/creating-a-genetic-algorithm-for-beginners/3

## xgboost parameter tuning

* http://xgboost.readthedocs.io/en/latest/parameter.html
* http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
