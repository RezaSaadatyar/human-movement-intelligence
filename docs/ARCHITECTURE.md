# Active Python code architecture

A minimal, understandable structure adapted from the maintainer's existing EEG and medical-LLM code conventions:

```text
code/python/src/core/
    data_loading.py
    data_inspection.py
studies/biomechanical_load_estimation/
    src/bonenet_data.py
    notebooks/01_data_loading_and_inspection.ipynb
tests/test_data_loading.py
requirements.txt
```

The general `load_data` function opens common file formats. `inspect_data` and `plot_signal` produce usable previews. The BoneNet adapter interprets its own MATLAB fields without embedding them in generic readers. The notebook imports these local functions directly; there is no custom HMI package installation.

After validating this stage in Cursor, the next objective is to prove reuse with HARTH (CSV), then incrementally develop preprocessing, subject-independent splitting, feature engineering and model training. No new function should be added without a demonstrated need and an executable output.
