"""Version information"""

__version__ = "0.0.1"
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401

from .fit_distribution import *
from .plotting import *
from .simulate import *
