"""
Utility functions for data preprocessing
"""
import pandas as pd
from pathlib import Path

def load_tabular_dataframe(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a tabular dataset from CSV file.
    
    Parameters
    ----------
    file_path : Path
        Path to the CSV file
    **kwargs
        Additional arguments to pass to pd.read_csv
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    return pd.read_csv(file_path, **kwargs)

__all__ = ['load_tabular_dataframe']
