"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
'''
from .adadelta import Adadelta  # noqa: F401
from .adagrad import Adagrad  # noqa: F401
from .adam import Adam  # noqa: F401
from .sparse_adam import SparseAdam  # noqa: F401
from .adamax import Adamax  # noqa: F401
from .asgd import ASGD  # noqa: F401
from .sgd import SGD  # noqa: F401
from .rprop import Rprop  # noqa: F401
from .rmsprop import RMSprop  # noqa: F401
from .optimizer import Optimizer  # noqa: F401
from .lbfgs import LBFGS  # noqa: F401
'''
from . import lr_scheduler  # noqa: F401

from .sgd_diagnostic_nonconvex import SGD_Diagnostic_Nonconvex
from .adam_diagnostic import Adam_Diagnostic
# from .sign_sgd import sign_SGD

'''
del adadelta
del adagrad
del adam
del sparse_adam
del adamax
del asgd
del sgd
del rprop
del rmsprop

del lbfgs
'''
del optimizer

del sgd_diagnostic_nonconvex
del adam_diagnostic
# del sign_sgd