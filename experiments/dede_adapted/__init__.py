# experiments/dede_adapted/__init__.py
"""
DeDe-Adapted: Masked Autoencoder for Network Traffic

Adapted from DeDe (CVPR 2025) for tabular network data.
"""

from .dede_model import (
    FeatureMasker,
    TabularEncoder,
    TabularDecoder,
    DeDeAdapted,
    build_dede_model
)

__all__ = [
    'FeatureMasker',
    'TabularEncoder', 
    'TabularDecoder',
    'DeDeAdapted',
    'build_dede_model'
]
