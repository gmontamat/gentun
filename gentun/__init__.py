# Make public APIs available at top-level import
from algorithms import GeneticAlgorithm
from populations import Population
from master import DistributedPopulation
from worker import GentunWorker

# Individual-Model pairs
try:
    from individuals import XgboostIndividual
    from models import XgboostModel
except ImportError:
    print("Warning: install xgboost to use XgboostIndividual and XgboostModel.")