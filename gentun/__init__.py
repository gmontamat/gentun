# Make public APIs available at top-level import
from .algorithms import GeneticAlgorithm, RussianRouletteGA
from .populations import Population, GridPopulation
from .server import DistributedPopulation, DistributedGridPopulation
from .client import GentunClient

# xgboost individuals and models
try:
    from .individuals import XgboostIndividual
    from .models.xgboost_models import XgboostModel
except ImportError:
    print("Warning: install xgboost to use XgboostIndividual and XgboostModel.")

# Keras individuals and models
try:
    from .individuals import GeneticCnnIndividual
    from .models.keras_models import GeneticCnnModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")

# Keras X0 individuals and models
try:
    from .individuals import GeneticCnnWithSkipIndividual
    from .models.genetic_cnn_with_skip_model import GeneticCnnWithSkipModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")
