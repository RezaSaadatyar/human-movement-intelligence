# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Read-only replay verification for historical HARTH Stage 25 protocols.

The original candidate, reconciled proposal and later approved qualified-Test
record are historical research evidence. Re-running Notebook 25 must not
rewrite these files, manufacture a new approval, or bypass frozen-code checks.
"""
import hashlib
import json
from pathlib import Path
import re


# ======================= 01. Research Protocol Lineage =======================
def verified_module_sha256(path, expected_sha256):
    """Require the approved source digest, allowing only Git checkout CRLF.

    This method never modifies the source file or its approved fingerprint.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("Frozen module SHA256 is malformed.")
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == expected_sha256:
        return expected_sha256
    if b"\r\n" in data:
        canonical = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if canonical == expected_sha256:
            return expected_sha256
    raise ValueError(
        "Frozen module SHA256 mismatch: " + str(path)
        + ". Do not overwrite the approved protocol. Review source changes."
    )



# Approved Stage 25 records retain original paths as historical evidence.
# Relocation changes where files live, never their approved hash or identity.
FROZEN_SOURCE_RELOCATIONS = {
    "code/python/src/core/advanced_temporal_models.py":
        "python_modules/core/advanced_temporal_models.py",
    "code/python/src/core/temporal_cnn.py":
        "python_modules/core/temporal_cnn.py",
    "code/python/src/core/temporal_benchmark.py":
        "python_modules/core/temporal_benchmark.py",
    "code/python/src/core/feature_engineering.py":
        "python_modules/core/feature_engineering.py",
    "code/python/src/core/robustness_experiments.py":
        "python_modules/core/robustness_experiments.py",
    "studies/wearable_activity_recognition/src/harth_data.py":
        "wearable_activity_recognition/code/src/harth_data.py",
}


def resolve_frozen_source_path(repository_root, recorded_relative):
    """Resolve one explicitly pinned legacy source path after relocation.

    The historical manifest key is retained; SHA256 verification remains
    mandatory and is performed separately on the unchanged source bytes.
    """
    if recorded_relative not in FROZEN_SOURCE_RELOCATIONS:
        raise ValueError(
            "Unrecognised Stage 25 historical source path: "
            + str(recorded_relative)
        )
    root = Path(repository_root).resolve()
    original = (root / recorded_relative).resolve()
    migrated = (root / FROZEN_SOURCE_RELOCATIONS[recorded_relative]).resolve()
    for candidate in (original, migrated):
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Frozen historical source missing at original and migrated paths: "
        + str(recorded_relative)
    )


def validate_stage25_lineage(candidate, original, reconciled, approved):
    """Compare a replay candidate with immutable original and approved records."""
    for name, obj in (
        ("candidate", candidate), ("original", original),
        ("reconciled", reconciled), ("approved", approved),
    ):
        if not isinstance(obj, dict) or obj.get("schema") != "harth-final-protocol-v1":
            raise ValueError(name + ": invalid frozen protocol schema.")
    if any(
        obj.get("status") != "PENDING_USER_REVIEW"
        for obj in (candidate, original, reconciled)
    ):
        raise ValueError("Historical Stage 25 candidates must remain pending.")
    if approved.get("status") != "APPROVED" or not isinstance(
        approved.get("approval_record"), dict
    ) or not approved["approval_record"]:
        raise ValueError("Existing approved evaluation provenance is missing.")

    # The historical original predates source-byte reconciliation. All the
    # methodological settings, datasets, saved notebook hashes and metrics
    # must agree; the original six module hashes remain immutable separately.
    shared_original = {
        key: value for key, value in original.items()
        if key != "module_sha256"
    }
    shared_reconciled = {
        key: value for key, value in reconciled.items()
        if key not in ("module_sha256", "reconciliation")
    }
    if shared_original != shared_reconciled:
        raise ValueError("Historical original differs in scientific protocol.")
    record = reconciled.get("reconciliation")
    if not isinstance(record, dict) or (
        record.get("original_module_sha256") != original.get("module_sha256")
    ):
        raise ValueError("Source reconciliation is not linked to the original.")

    expected_candidate = {
        key: value for key, value in reconciled.items()
        if key != "reconciliation"
    }
    if candidate != expected_candidate:
        different = sorted(
            key for key in set(candidate) | set(expected_candidate)
            if candidate.get(key) != expected_candidate.get(key)
        )
        raise ValueError(
            "Replayed Stage 25 proposal differs from the reviewed protocol in "
            + ", ".join(different)
            + ". Do not overwrite historical manifests."
        )

    recorded_approval = {
        key: value for key, value in approved.items()
        if key not in ("status", "approval_record")
    }
    recorded_pending = {
        key: value for key, value in reconciled.items()
        if key != "status"
    }
    if recorded_approval != recorded_pending:
        raise ValueError("Approved manifest differs from reconciled protocol.")

    return {
        "original_proposal": "PRESERVED",
        "reconciled_proposal": "MATCHED",
        "existing_approval": "VERIFIED_EXISTING_ONLY",
        "new_approval_granted": False,
        "files_written": 0,
    }


def verify_preexisting_stage25_lineage(candidate, evidence_dir):
    """Validate exact lineage from existing files, without writing anything."""
    folder = Path(evidence_dir)
    filenames = (
        "stage_25_protocol_proposal.json",
        "stage_25_protocol_proposal_reconciled.json",
        "stage_25_protocol_approved_qualified_test.json",
    )
    sources = []
    for name in filenames:
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(
                "Missing original Stage 25 provenance record: " + str(path)
            )
        sources.append(json.loads(path.read_text(encoding="utf-8")))
    return validate_stage25_lineage(candidate, *sources)
