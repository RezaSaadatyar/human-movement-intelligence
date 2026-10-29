# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Reusable data summaries and visualisations for scientific arrays and tables."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================ 01. Dataset Summary ============================
def inspect_data(data, name='Dataset', preview_rows=5):
    """
    Display a concise structural summary of a table or numeric array.

    Args:
        data (pd.DataFrame | np.ndarray): Decoded dataset or signal array.
        name (str): Name shown in the output.
        preview_rows (int): Maximum number of tabular rows to show.

    Returns:
        dict: Shape, type and missing/nonfinite counts for further auditing.
    """
    print(f'\n===== {name} =====')
    if isinstance(data, pd.DataFrame):
        report = {
            'name': name, 'shape': tuple(data.shape),
            'columns': list(data.columns),
            'missing_values': int(data.isna().sum().sum()),
            'dtypes': {str(k): str(v) for k, v in data.dtypes.items()},
        }
        print(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
        print(f"Missing values: {report['missing_values']}")
        print('Columns:', ', '.join(map(str, data.columns)))
        print(data.head(preview_rows).to_string(index=False))
        return report

    array = np.asarray(data)
    if array.ndim == 0:
        raise ValueError('Expected an array with a sample dimension.')
    nonfinite = (
        int(np.count_nonzero(~np.isfinite(array)))
        if np.issubdtype(array.dtype, np.number) else None
    )
    report = {'name': name, 'shape': tuple(array.shape),
              'dtype': str(array.dtype), 'nonfinite_values': nonfinite}
    print(f'Shape: {array.shape} | Data type: {array.dtype}')
    print(f'NaN/Inf count: {nonfinite if nonfinite is not None else "not numeric"}')
    print('Preview of first observation:')
    print(np.array2string(array[0], threshold=12, edgeitems=2))
    return report


# ========================= 02. Signal Visualisation ==========================
def plot_signal(signal, title='Example signal', channel_names=None):
    """
    Plot one 2D (channels, time) signal or a 1D sequence.

    Args:
        signal (np.ndarray): Shape (channels, time) or (time,).
        title (str): Figure title.
        channel_names (list[str] | None): Labels for plotted channels.

    Returns:
        matplotlib.figure.Figure: Figure for display or saving.
    """
    values = np.asarray(signal)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError('Provide one record with shape (channels, time).')
    if channel_names is not None and len(channel_names) != values.shape[0]:
        raise ValueError('channel_names must match the number of channels.')
    fig, axis = plt.subplots(figsize=(10, 3.5))
    for channel in range(values.shape[0]):
        label = channel_names[channel] if channel_names else f'Channel {channel + 1}'
        axis.plot(values[channel], label=label)
    axis.set(title=title, xlabel='Sample / normalised time index', ylabel='Source value')
    if values.shape[0] <= 8:
        axis.legend(loc='best')
    fig.tight_layout()
    return fig
