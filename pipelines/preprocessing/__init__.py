"""
Preprocessing pipelines for various IDS datasets.
"""

from .cicids2017 import preprocess_cicids2017, PreprocessResult as PreprocessResult2017
from .cicids2018 import preprocess_cicids2018, PreprocessResult as PreprocessResult2018

__all__ = [
    "preprocess_cicids2017",
    "preprocess_cicids2018",
    "PreprocessResult2017",
    "PreprocessResult2018",
]
