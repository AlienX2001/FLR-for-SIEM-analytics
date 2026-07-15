# Federated Logistic Regression Prototype

This project is a research prototype for federated SIEM log classification with PRF-aligned vocabularies and a label-conditioned hierarchical ensemble.

It has two independent code paths:

- `federated_lr_pipeline/`: steps 1-5, covering local vocabulary extraction, PRF tagging, global PRF-tag vocabulary construction, one-vs-rest specialist LR training, plaintext federated parameter aggregation, label-level logit fusion, and inference.
- `ioc_generation/`: step 6, covering high-risk prediction explanation correlation, log normalization, IoC extraction, and STIX v2 JSON template generation.

IoC generation is separate from training and inference. It consumes row-aligned prediction outputs and re-reads the original log CSVs by `org_index` and `row_index`.

## Architecture

The default model is a label-conditioned hierarchical ensemble. For each class label, the pipeline trains binary one-vs-rest specialists for the configured subcategories.

Example:

```text
credential_attack:
  system
  network
  identity
  cross

data_exfiltration:
  system
  network
  llm
  cross
```

Each `(label, subcategory)` specialist has its own:

- local vocabulary per organization
- PRF-tagged vocabulary
- global PRF-tag vocabulary
- index mapping
- binary LR weights and bias
- local training loop
- per-round metrics

For a label `y`, specialist logits are fused before softmax:

```text
z_y = beta_y_bias + sum_s beta_y_s * z_y_s
```

Final class probabilities are:

```text
softmax([z_benign, z_credential_attack, z_data_exfiltration, ...])
```

The pipeline never averages probabilities for the final decision.

## Vocabulary Alignment

Organizations generate local vocabularies. Each token is tagged with HMAC-SHA256 using a namespace:

```text
tag = PRF(k, label + "|" + subcategory + "|" + canonical_token)
```

The server builds global vocabularies from PRF tags only:

```text
GV[label][subcategory] = sorted union of tags
```

Plaintext tokens are not needed by the global vocabulary builder. Use `--debug-plaintext-vocab` only for local debugging and explanation inspection.

## Cross Features

Cross-category features use a fixed vocabulary before training. Examples:

- `cross:sensitive_file_read_AND_large_upload_same_host_15m`
- `cross:encoded_command_AND_first_seen_domain_same_host_15m`
- `cross:llm_file_read_tool_AND_system_sensitive_file_read_same_user_15m`
- `cross:failed_login_burst_AND_successful_login_same_user_15m`
- `cross:secret_read_tool_AND_external_post_same_user_15m`

These features are generated from primitive signals before training and are PRF-tagged with the same label/subcategory namespace.

## Input CSV Format

Each organization provides:

- one data CSV
- one row-aligned groundtruth CSV

The n-th row of the data CSV corresponds to the n-th row of the groundtruth CSV. The pipeline does not merge by ID and does not require `log_id`.

Each row receives:

```text
internal_log_id = org_{org_index}_row_{row_index}
```

If a source `log_id`-like column exists, it is preserved as metadata only.

## Example Commands

Running LR training:

```bash
conda run -n LR python -m federated_lr_pipeline.run \
  --org-data examples/orgA_logs.csv examples/orgB_logs.csv examples/orgC_logs.csv \
  --org-groundtruth examples/orgA_groundtruth.csv examples/orgB_groundtruth.csv examples/orgC_groundtruth.csv \
  --num-features 20 \
  --federation-iterations 5 \
  --min-df 1 \
  --max-df 1.0 \
  --risk-threshold 0.5 \
  --test-size 0.2 \
  --class-weight balanced \
  --aggregation-weighting sample_size \
  --output-dir outputs/run_001
```

Optional hierarchy config:

```bash
--hierarchical-config path/to/hierarchy.json
```

The JSON can define labels, subcategories, and manual fusion weights.

Run separate IoC generation:

```bash
conda run -n LR python -m ioc_generation.run \
  --high-risk-logs outputs/run_001/high_risk_logs.jsonl \
  --explanations outputs/run_001/explanations.jsonl \
  --org-data examples/orgA_logs.csv examples/orgB_logs.csv examples/orgC_logs.csv \
  --output-dir outputs/run_001/iocs
```

Run inference-only testing from a previously trained artifact directory:

```bash
conda run -n LR python -m federated_lr_pipeline.run \
  --testing \
  --org-data examples/orgA_logs.csv examples/orgB_logs.csv examples/orgC_logs.csv \
  --org-groundtruth examples/orgA_groundtruth.csv examples/orgB_groundtruth.csv examples/orgC_groundtruth.csv \
  --model-artifact-dir outputs/run_001 \
  --risk-threshold 0.5 \
  --output-dir outputs/test_run_001
```

Testing mode loads saved specialist weights, global PRF-tag vocabularies, per-organization index vectors, label classes, and fusion metadata. It does not regenerate vocabularies, train local models, aggregate parameters, initialize new models, or write new trained weights. Testing from raw logs requires saved local vocabulary tokens; run the training job with `--debug-plaintext-vocab` when you need later standalone testing from CSV inputs.

## Output Files

LR training outputs:

- `run_config.json`
- `hierarchical_config.json`
- `hierarchical_model_manifest.json`
- `manual_logit_fusion.json`
- `label_encoder_classes.json`
- `training_metrics.json`
- `predictions.csv`
- `predictions.jsonl`
- `high_risk_logs.jsonl`
- `explanations.jsonl`

Specialist artifacts are written per label/subcategory, for example:

- `data_exfiltration_network_global_vocabulary_tags.json`
- `final_data_exfiltration_network_weights.npy`
- `final_data_exfiltration_network_bias.npy`
- `org_0_data_exfiltration_network_lv_tags.json`
- `org_0_data_exfiltration_network_gv_index_vector.json`

When `--debug-plaintext-vocab` is enabled, local debug token files are also written.

Testing mode writes only testing outputs:

- `testing_run_config.json`
- `testing_metrics.json`
- `testing_classification_report.txt`
- `testing_confusion_matrix.csv`
- `testing_predictions.csv`
- `testing_predictions.jsonl`
- `testing_high_risk_logs.jsonl`
- `testing_explanations.jsonl`

IoC-generation writes:

- `ioc_bundle.json`
- `ioc_records.jsonl`
- `ioc_summary.csv`
