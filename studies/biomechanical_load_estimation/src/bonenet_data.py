# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Read the authors' processed BoneNet IMU and FE-strain arrays with Python."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from core.data_loading import load_data


SOURCE = (
    'https://raw.githubusercontent.com/ZainabAltai/'
    'BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-'
    'Femoral-Neck-strains-from-Exercise-Data/main/'
)


# ============================== 1. Data Acquisition ==============================
def ensure_bonenet_files(data_dir):
    """
    Download two public source files only when absent locally.

    Args:
        data_dir (str | Path): Local raw-data folder, ignored by Git.

    Returns:
        Path: Directory containing Data.mat and labels.csv.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename in ('Data.mat', 'labels.csv'):
        path = root / filename
        if not path.exists():
            print(f'Downloading {filename} from the public authors repository...')
            partial = root / f'{filename}.part'
            try:
                urlretrieve(SOURCE + filename, partial)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        elif not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Invalid cached source file: {path}')
    return root


# ================================ 2. Data Loading ================================
def load_bonenet_data(data_dir):
    """
    Decode numeric IMU trajectories and model-derived FE targets.

    Args:
        data_dir (str | Path): Folder with Data.mat and labels.csv.

    Returns:
        X (np.ndarray): (records, channels, time).
        y (np.ndarray): (records, time, FE targets).
        labels (pd.DataFrame): Authors' subject/activity/trial CSV rows.
        feature_names (list[str]): Input channel names.
        target_names (list[str]): FE target names, usually e1 and e3.

    Notes:
        Dataset-specific MATLAB field handling belongs here, not inside
        the common file reader. The label row-order linkage has not been
        independently verified, and reported units require source review.
    """
    root = Path(data_dir)
    mat = load_data(root / 'Data.mat')
    labels = load_data(root / 'labels.csv')
    required = {'SubjID', 'Task', 'TrialNum'}
    if not required.issubset(labels.columns):
        raise ValueError(f'Expected label columns: {sorted(required)}')
    inputs, outputs = mat.get('inputs'), mat.get('outputs')
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError('BoneNet MAT file needs numeric inputs and outputs structs.')

    channels = []
    feature_names = []
    for axis in ('x', 'y', 'z'):
        if axis not in inputs or not isinstance(inputs[axis], dict):
            raise ValueError(f'Missing numeric IMU axis: {axis}')
        for name, values in inputs[axis].items():
            channel = np.asarray(values)
            if channel.ndim != 2 or not np.issubdtype(channel.dtype, np.number):
                raise ValueError(f'Invalid IMU channel {axis}.{name}: {channel.shape}')
            channels.append(channel)
            feature_names.append(f'{axis}.{name}')
    if len({item.shape for item in channels}) != 1:
        raise ValueError('IMU channels do not share the same (record, time) shape.')
    X = np.stack(channels, axis=1)

    target_names = list(outputs)
    target_channels = [np.asarray(outputs[name]) for name in target_names]
    if not target_channels or any(
        item.ndim != 2 or not np.issubdtype(item.dtype, np.number)
        for item in target_channels
    ):
        raise ValueError('Targets must be 2D numeric trajectory arrays.')
    if len({item.shape for item in target_channels}) != 1:
        raise ValueError('Target shapes are inconsistent.')
    y = np.stack(target_channels, axis=-1)
    if X.shape[0] != y.shape[0] or X.shape[0] != len(labels):
        raise ValueError('Inputs, outputs and label rows have different lengths.')
    print(f'IMU X: {X.shape} | FE targets y: {y.shape}')
    print(f'Subject identifiers in labels: {labels.SubjID.nunique()}')
    return X, y, labels, feature_names, target_names
