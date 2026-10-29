# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Read common scientific file formats without assuming dataset-specific labels."""

from pathlib import Path

import pandas as pd
from scipy.io import loadmat


# ============================= 01. File Loading ==============================
def load_data(file_path, **kwargs):
    """
    Load CSV, Excel, delimited text or standard MATLAB MAT files.

    Args:
        file_path (str | Path): Path to the local source file.
        **kwargs: Optional pandas read parameters for tabular formats.
            For .txt/.dat, whitespace separation and no header are defaults.

    Returns:
        pd.DataFrame: For .csv, .xlsx, .xls, .txt and .dat files.
        dict: For standard MAT files (variable names mapped to arrays/structs).

    Notes:
        A file reader does not infer subject IDs, targets or sensor units.
        MATLAB object classes may require a validated dataset-specific adapter.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    extension = path.suffix.lower()
    if extension == '.csv':
        return pd.read_csv(path, **kwargs)
    if extension in ('.xlsx', '.xls'):
        return pd.read_excel(path, **kwargs)
    if extension in ('.txt', '.dat'):
        options = {'sep': r'\s+', 'header': None}
        options.update(kwargs)
        return pd.read_csv(path, **options)
    if extension == '.mat':
        if kwargs:
            raise ValueError('MAT files do not accept pandas read options.')
        try:
            return loadmat(path, simplify_cells=True)
        except NotImplementedError as exc:
            raise ValueError('MAT v7.3/HDF5 requires a separate compatible reader.') from exc

    raise ValueError(
        f"Unsupported format '{extension}'. Supported: CSV, XLSX, XLS, TXT, DAT, MAT."
    )
