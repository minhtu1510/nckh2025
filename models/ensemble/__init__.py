"""
Ensemble Learning Models for IDS Defense

Available ensembles:
- StackingEnsemble: Meta-learning based combination
"""

from .stacking import StackingEnsemble, create_stacking_ensemble

__all__ = ['StackingEnsemble', 'create_stacking_ensemble']
