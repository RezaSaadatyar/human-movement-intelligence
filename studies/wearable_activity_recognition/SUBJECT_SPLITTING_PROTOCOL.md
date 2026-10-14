# HARTH group-independent splitting

**Purpose:** Verify reusable train/validation/test splitting without leakage from overlapping windows or repeated participant IDs.

- Eight selected HARTH v2.0 participants: S023, S024, S025, S027, S028, S032, S034, S035.
- Data are from the pinned NTNU authors' repository. Combined source files are approximately 124 MiB, stored only locally.
- Same 100-point windows, 50-point stride and 100% label purity as prior notebooks.
- Five deterministic time-domain signal features per channel; 30 columns for six accelerometers.
- **Split:** 4 independent Train participants, 2 Validation, 2 Test; random seed 42.
- Fractions are per participant. Row/window counts may differ. No stratification guarantee exists.
- All overlaps are checked at the subject-ID level and participant windows stay together.
- Notebook 04 displays subject memberships, class counts and missing classes by partition.
- No preprocessing is fitted and no model is trained. Full-benchmark subject-wise cross-validation remains future work.

**Code:** `code/python/src/core/data_splitting.py` for generic grouping; `studies/wearable_activity_recognition/src/harth_data.py` remains dataset-specific.
