# Experiment Results

## Experimental Setting

The experiment used a three-organization split of the CICAPT-IIoT dataset with row-aligned log and groundtruth files. The supervised target column was `Tactic Name`. The model was trained with a two-level categorization structure: the first level was the tactic label, and the second level was the evidence subcategory used by each label-specific specialist.

The trained taxonomy contained nine tactic labels and 29 label-subcategory specialists:

| Tactic label | Evidence subcategories |
|---|---|
| Benign | system, network, cross |
| cleanup | system, network, cross |
| collection | system, network, cross |
| command and control | system, network, cross |
| credential access | system, network, identity, cross |
| discovery | system, network, cross |
| exfiltration | system, network, llm, cross |
| lateral movement | system, network, cross |
| persistence | system, network, cross |

The training configuration used 1,000 local features per specialist, five federation rounds, ten local epochs per round, balanced class weighting, sample-size-weighted aggregation, and a risk threshold of 0.75. Vocabulary construction used training rows only, with `min_df=2` and `max_df=0.95`.

The internal training split contained 830,975 training rows and 207,745 held-out rows. The separate testing run evaluated 593,188 labeled rows.

## Training Results

The model improved sharply after the first federation round. The first round used TF-IDF-style features, while subsequent rounds used term-frequency features. Held-out accuracy increased from 87.75% after round 1 to 93.09% after round 5. Performance stabilized after the second round, indicating that most measurable gains were obtained early in federated training.

| Round | Feature mode | Held-out accuracy | Macro F1 | Weighted F1 |
|---:|---|---:|---:|---:|
| 1 | TF-IDF | 0.8775 | 0.8642 | 0.8620 |
| 2 | TF | 0.9307 | 0.9327 | 0.9306 |
| 3 | TF | 0.9305 | 0.9324 | 0.9303 |
| 4 | TF | 0.9309 | 0.9330 | 0.9308 |
| 5 | TF | 0.9309 | 0.9330 | 0.9308 |

Final-round accuracy was consistent across organizations:

| Organization | Held-out rows | Final held-out accuracy |
|---:|---:|---:|
| 0 | 69,112 | 0.9311 |
| 1 | 69,263 | 0.9304 |
| 2 | 69,370 | 0.9312 |

This consistency suggests that the federated aggregation procedure produced a model with similar held-out behavior across the three participating organizations rather than overfitting to a single organization.

## Testing Results

On the separate testing split, the final ensemble achieved 93.83% accuracy, 92.77% macro F1, and 93.85% weighted F1 over 593,188 labeled rows.

| Metric | Value |
|---|---:|
| Accuracy | 0.9383 |
| Macro precision | 0.9260 |
| Macro recall | 0.9333 |
| Macro F1 | 0.9277 |
| Weighted precision | 0.9417 |
| Weighted recall | 0.9383 |
| Weighted F1 | 0.9385 |

Per-class performance was strongest for cleanup, exfiltration, lateral movement, persistence, and Benign traffic. Credential access was the most difficult class, with substantially lower recall and F1 than the other categories.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Benign | 0.9019 | 1.0000 | 0.9484 | 15,306 |
| cleanup | 0.9947 | 1.0000 | 0.9974 | 88,423 |
| collection | 1.0000 | 0.9211 | 0.9589 | 88,300 |
| command and control | 0.7935 | 0.9193 | 0.8517 | 56,139 |
| credential access | 0.7700 | 0.7623 | 0.7661 | 55,765 |
| discovery | 0.9834 | 0.8225 | 0.8958 | 56,197 |
| exfiltration | 0.9817 | 0.9914 | 0.9865 | 88,387 |
| lateral movement | 0.9844 | 0.9947 | 0.9895 | 88,300 |
| persistence | 0.9245 | 0.9883 | 0.9553 | 56,371 |

The main residual errors were concentrated among attack classes with overlapping behavioral signatures. The largest confusion was between credential access and command and control, followed by discovery errors into credential access, command and control, and persistence.

| True class | Predicted class | Count |
|---|---|---:|
| credential access | command and control | 9,945 |
| command and control | credential access | 4,532 |
| discovery | credential access | 3,830 |
| collection | credential access | 3,673 |
| discovery | command and control | 2,726 |
| credential access | persistence | 2,530 |
| discovery | persistence | 2,022 |
| collection | Benign | 1,660 |
| collection | exfiltration | 1,631 |
| discovery | lateral movement | 1,395 |

The diagnostic single-view subcategory results show that system and network evidence carried nearly all of the standalone classification power. The cross, identity, and LLM views were much weaker when evaluated as isolated views, although they still participate in the final fused ensemble for the labels where they are configured.

| Evidence view | Standalone diagnostic accuracy |
|---|---:|
| system | 0.9407 |
| network | 0.9388 |
| llm | 0.1742 |
| identity | 0.1196 |
| cross | 0.0950 |

The primary testing run marked 578,708 of 593,188 rows as high-risk at the 0.75 threshold, corresponding to 97.56% of the testing rows. The IOC-specific testing run contained 516,873 predictions and 410,597 high-risk rows, corresponding to 79.44% of that run.

## IOC Generation and Validation Results

The IOC generation run produced 1,008,095 occurrence-level IOC records. These occurrence counts are larger than the unique comparison counts because the same indicator can appear in multiple high-risk events.

| Indicator type | Occurrence count |
|---|---:|
| IPv4 | 657,464 |
| Domain | 299,533 |
| URL | 50,768 |
| MD5 | 315 |
| SHA1 | 15 |

IOC validation compared normalized unique indicators extracted from the generated STIX-style output against normalized unique indicators from the ground truth. The comparison procedure parses STIX indicator patterns, normalizes values by type, expands URL values to comparable host indicators where appropriate, and computes set intersections and differences per indicator type.

Across all IOC types, the generated output contained 152,959 unique normalized indicators, the ground truth contained 168,207 unique normalized indicators, and 152,642 indicators overlapped. This corresponds to 99.79% micro-precision, 90.75% micro-recall, and 95.05% micro-F1.

| Indicator type | Generated unique | Ground truth unique | Intersection | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Domain | 125,795 | 137,461 | 125,705 | 0.9993 | 0.9145 | 0.9550 |
| IPv4 | 3,714 | 4,037 | 3,655 | 0.9841 | 0.9054 | 0.9431 |
| URL | 23,371 | 26,705 | 23,282 | 0.9962 | 0.8718 | 0.9299 |
| MD5 | 73 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 |
| SHA1 | 6 | 0 | 0 | 0.0000 | n/a | 0.0000 |
| SHA256 | 0 | 3 | 0 | n/a | 0.0000 | 0.0000 |
| **Total** | **152,959** | **168,207** | **152,642** | **0.9979** | **0.9075** | **0.9505** |

The IOC results indicate that the pipeline recovered most high-volume network observables with very high precision. Domain, IPv4, and URL extraction all exceeded 98% precision. Recall was lower than precision, especially for URLs, indicating that the generated set was conservative relative to the ground truth. The low hash performance reflects a mismatch between generated and ground-truth hash values: MD5, SHA1, and SHA256 categories had no overlapping values in the provided comparison.

Overall, the experiment demonstrates strong classification performance and high-precision IOC extraction for network observables. The main limitations observed in the results are reduced classification performance for credential access and discovery, high high-risk selection rates at the configured threshold, and poor agreement for hash-based IOCs.
