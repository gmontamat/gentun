# gentun: genetic algorithm hyerparameter tuning

The purpose of this project is to provide a simple framework for hyperparameter tuning of machine learning models
such as Neural Networks and Gradient Boosted Trees using a genetic algorithm. Since measuring the fitness of a
particular individual of a given population is time consuming, a client-server approach is used to allow several
clients perform model fitting and cross-validation of the individuals that a server assigns. Reproduction and
mutation of the population is handled by the server.

# Supported models

* xgboost

# References

* Artificial Intelligence: A Modern Approach. 3rd edition. Chapter 4.
