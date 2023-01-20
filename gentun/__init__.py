# Make public APIs available at top-level import
from .populations import Population, GridPopulation
from .server import DistributedPopulation, DistributedGridPopulation
from .client import GentunClient

# Genetic Algorithms
try:
    from .genetic_algorithms.genetic_algorithm import GeneticAlgorithm
    from .genetic_algorithms.russian_roulette_genetic_algorithm import RussianRouletteGA
    from .genetic_algorithms.nsga_net import NSGANet
except ImportError:
    print("Warning: install genetic algorithms to use GeneticAlgorithm, RussianRouletteGA and NSGANet.")

# xgboost individuals and models
try:
    from .individuals.xgboost_individual import XgboostIndividual
    from .models.xgboost_models import XgboostModel
except ImportError:
    print("Warning: install xgboost to use XgboostIndividual and XgboostModel.")

# Keras individuals and models
try:
    from .individuals.genetic_cnn_individual import GeneticCnnIndividual
    from .models.keras_models import GeneticCnnModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")

# Keras X0 individuals and models
try:
    from .individuals.genetic_cnn_with_skip_individual import GeneticCnnWithSkipIndividual
    from .models.genetic_cnn_with_skip_model import GeneticCnnWithSkipModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")
