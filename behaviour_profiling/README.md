# Behaviour Profiling

`behaviour_profiling` is a standalone benign-behaviour baseline and novelty
detector for CSV log streams. It does not train or invoke the federated LR
pipeline, policy filter, IoC generator, or an attack classifier.

The module learns what was observed in explicitly benign training rows and
flags later rows that fall outside that profile. A detected anomaly means
"different from the benign baseline"; it does not prove malicious activity.

## Commands

Train from CSV files that already contain only benign events:

```bash
python -m behaviour_profiling.cli train \
  --logs data/org_0_benign.csv data/org_1_benign.csv \
  --config behaviour_profiling/examples/profile_config.yaml \
  --output outputs/behaviour/profile.json
```

Train from mixed logs using row-aligned ground truth:

```bash
python -m behaviour_profiling.cli train \
  --logs data/org_0_logs.csv data/org_1_logs.csv \
  --groundtruth data/org_0_groundtruth.csv data/org_1_groundtruth.csv \
  --label-column "Tactic Name" \
  --benign-label Benign \
  --config behaviour_profiling/examples/profile_config.yaml \
  --output outputs/behaviour/profile.json
```

The n-th ground-truth row is matched with the n-th log row. IDs are not used
for alignment. If `--groundtruth` is omitted, every input row is treated as
benign training data.

Detect anomalies:

```bash
python -m behaviour_profiling.cli detect \
  --logs data/new_logs.csv \
  --profile outputs/behaviour/profile.json \
  --output outputs/behaviour/anomalies.csv \
  --results-jsonl outputs/behaviour/all_results.jsonl
```

`anomalies.csv` contains only anomalous rows. `--results-jsonl` is optional and
contains the decision for every row.

Common options are `--encoding`, `--delimiter`, and `--log-level`.

## Shared Preprocessing

[`preprocessing_adapter.py`](preprocessing_adapter.py) calls the federated
pipeline's existing missing-value, canonical-value, and field-aware token
functions. It does not fit a vectorizer, construct a vocabulary, generate PRF
tags, train LR models, aggregate parameters, or run inference. Raw CSV values
are preserved in anomaly output.

## Profile Contents

The profile is a versioned, human-readable JSON artifact containing:

- categorical values observed often enough in benign data;
- numeric count, minimum, maximum, mean, and standard deviation;
- normalized field-aware text tokens observed in benign data;
- an optional profile for each sufficiently represented entity;
- training counts, field types, and the complete detection configuration.

The artifact is written atomically and does not use pickle or executable model
serialization.

## Configuration

The example at
[`examples/profile_config.yaml`](examples/profile_config.yaml) documents every
setting. YAML and JSON are supported.

`fields.entity` defines the columns identifying an entity, such as `host` or
`user_uid`. Entity profiles are used when they have at least
`training.minimum_entity_rows`; otherwise detection uses the global profile.
An unseen or insufficiently trained entity can contribute `UNKNOWN_ENTITY`.

`fields.categorical` uses exact canonical values. `fields.numeric` learns
benign numeric moments and ranges. `fields.text` uses field-aware tokens.
`fields.ignored` excludes labels, timestamps, and other values that must not
affect the profile.

When `fields.auto_infer` is enabled, unconfigured nonempty fields are inferred:

- sufficiently numeric fields become numeric profiles;
- bounded-cardinality fields become categorical profiles;
- high-cardinality fields become token-based text profiles.

Timestamp-like fields are ignored during automatic inference by default so a
later timestamp does not make every new event anomalous. Administrators can
explicitly configure a timestamp-derived field if it should be profiled.

## Detection and Scoring

Each deviation contributes a configurable score:

- `UNKNOWN_ENTITY`
- `MISSING_PROFILED_FIELD`
- `UNKNOWN_CATEGORICAL_VALUE`
- `NUMERIC_VALUE_MALFORMED`
- `NUMERIC_OUTSIDE_BENIGN_PROFILE`
- `UNSEEN_TEXT_TOKENS`
- `MALFORMED_ROW`
- `INSUFFICIENT_PROFILE_EVIDENCE`

A row is anomalous when the sum reaches
`detection.anomaly_score_threshold`. Numeric checks can enforce the complete
observed range, a standard-deviation threshold, or both. Text checks compare
the fraction of normalized tokens absent from benign training.
Missingness contributes an anomaly only when missing-field checks are enabled
and the selected benign profile never observed that field missing.

The anomaly output places these metadata columns before the original fields:

- `source_file`
- `source_row_number`
- `source_line_number`
- `behaviour_action`
- `anomaly_score`
- `profile_scope`
- `anomaly_reasons`
- `anomaly_reason_details`

CSV records containing embedded newlines retain the physical line where the
record started. Output is written atomically.

## Operational Guidance

Train with representative benign data covering normal shifts, maintenance,
software updates, and expected user/network variation. A narrow profile will
produce excessive anomalies. A profile trained on contaminated data may learn
malicious behaviour as normal.

Numeric observed-range enforcement is intentionally strict. Disable
`enforce_observed_numeric_range` and rely on the configured standard-deviation
threshold when benign measurements naturally extend beyond historical extrema.

Do not include ground-truth labels or attack annotations as profile features.
The example configuration excludes them. Profiles should be retrained or
versioned when infrastructure and normal behaviour change.
