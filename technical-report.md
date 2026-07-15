# Approach

This project presents a federated experimental framework for security-log classification and subsequent indicator-of-compromise generation. The central objective is to evaluate how multiple organizations can participate in a shared threat-classification workflow while limiting direct exposure of local feature vocabularies.

The framework consists of two sequential analytical stages. The first stage performs federated supervised classification over organization-specific security logs. The second stage uses high-risk classification outputs to derive concrete cyber-threat observables, such as IP addresses, domains, URLs, and cryptographic file hashes.

The classification design is based on a privacy-preserving vocabulary-alignment mechanism. Each organization independently derives features from its local logs. These local features are transformed into deterministic cryptographic tags before global alignment. The resulting shared representation permits cross-organization feature coordination without requiring the central process to construct its global vocabulary from plaintext feature names.

The model architecture is hierarchical and label-conditioned. Each target class is represented by a set of specialist models corresponding to distinct evidence categories, including network, system, identity, LLM/tool, cloud, and cross-category evidence. These specialists model different semantic aspects of the same event. Their outputs are combined through label-specific score fusion to produce final class probabilities.

The resulting taxonomy is a two-level categorization scheme. The first level is the class label, which represents the overall security outcome assigned to an event. The second level is the evidence subcategory, which represents the analytical perspective used to evaluate that label, such as network behavior, system behavior, identity behavior, LLM/tool behavior, cloud behavior, or cross-category behavior. Thus, the model does not only ask which label an event belongs to; it also organizes the supporting evidence by the type of telemetry from which that label-specific evidence is derived.

The downstream IOC stage operationalizes the classification results. Rather than treating the model prediction as the final artifact, the pipeline uses high-risk rows as candidates for observable extraction. The original log rows are revisited, normalized, and searched for validated indicators. The resulting indicators are emitted in structured form so that model outputs can be connected to analyst-facing threat intelligence artifacts.

The conceptual pipeline is:

```text
organization-specific logs
        |
        v
field-aware feature construction
        |
        v
privacy-preserving vocabulary alignment
        |
        v
federated specialist learning
        |
        v
hierarchical score fusion
        |
        v
high-risk event selection
        |
        v
indicator extraction from original evidence
        |
        v
structured IOC artifacts
```

# End-to-end experiment procedure

The experiment begins with multiple organizations contributing paired datasets: security event logs and corresponding groundtruth labels. The pairing is positional; each log row is associated with the label row at the same position. This row-level alignment is maintained throughout the experiment by assigning an internal row identity to every event.

The procedure can be described as five stages.

First, the input data are loaded, validated, and indexed. Each organization is treated as a separate participant. The system verifies that every organization has a valid log-label pairing and assigns stable row identifiers that persist into prediction and IOC artifacts.

Second, labels are encoded into a shared class space. This ensures that all organizations use a common interpretation of class membership even though their log contents remain organization-specific.

Third, raw events are transformed into multiple evidence views. Instead of constructing one undifferentiated feature representation, the experiment decomposes each event into semantically motivated subcategory representations. This supports separate modeling of network behavior, host behavior, identity behavior, LLM/tool behavior, cloud behavior, and cross-category patterns.

Fourth, federated training is performed over the specialist models. Each organization trains local updates over its own feature representation. These updates are aggregated into global specialist parameters. This process is repeated for the configured number of federation rounds, with evaluation performed on held-out rows after each round.

Fifth, the trained ensemble is used to score all available rows. Specialist scores are fused into final label probabilities. Rows that exceed the risk threshold are retained as high-risk events and are accompanied by explanation-oriented metadata describing influential model evidence.

The IOC generation stage is then applied to the high-risk event set. Each high-risk record points back to the original organization and row. The original event text is normalized, searched for supported observables, and converted into structured IOC records. The IOC stage therefore serves as a post-classification transformation from probabilistic risk assessment to concrete threat indicators.

In abstract form, the experiment proceeds as:

```text
labeled organizational logs
-> semantic feature views
-> tagged vocabulary alignment
-> federated specialist training
-> fused risk prediction
-> high-risk event selection
-> IOC extraction
```

The classification and IOC stages are analytically distinct. Classification estimates the likelihood that an event belongs to a security-relevant class. IOC generation extracts concrete observables from events already selected by the classifier.

## Data source

The primary data source is a collection of organization-specific CSV pairs. Each pair consists of an event-log table and a corresponding label table. The experimental design assumes row-aligned supervision: the label for an event is obtained from the label table row at the same ordinal position.

The event logs may contain heterogeneous security telemetry, including network attributes, process metadata, file-system attributes, user or identity fields, cloud-context fields, and LLM/tool-use fields. This heterogeneity motivates the use of subcategory-specific feature views rather than a single uniform feature representation.

The groundtruth labels define the supervised learning target. Labels may represent benign behavior or security-relevant activity categories. During training, labels are mapped into a shared class space so that all organizations contribute to a common classification task.

Each event is assigned an internal row identity consisting of its organization index and row index. If the source data contain an external event identifier, that identifier may be preserved as metadata; however, the experimental alignment mechanism remains positional.

Conceptually, each event contributes:

- Raw event attributes.
- A groundtruth label.
- A persistent internal row identity.
- Optional source-system metadata.
- Derived feature representations for each applicable evidence category.

## Data processing & generation

The preprocessing stage converts heterogeneous raw logs into structured representations suitable for federated learning and later IOC extraction.

The first transformation is row normalization and identity preservation. Each event is loaded with its corresponding label, assigned an internal identifier, and retained in an organization-specific context. This enables subsequent outputs, including predictions and indicators, to be traced back to the originating event.

The second transformation is semantic feature construction. Event attributes are grouped into evidence categories. A network representation emphasizes communication attributes; a system representation emphasizes process and file behavior; identity, cloud, and LLM/tool representations emphasize their respective fields; and cross-category features capture selected combinations of suspicious conditions within the same event.

The feature construction strategy is field-aware. Tokens are not treated merely as isolated strings; they are associated with the field in which they occur. This preserves contextual meaning. For example, the same textual value can carry different significance when it appears in a process path, a destination address, or a user identifier.

The third transformation is local vocabulary construction. Each organization derives vocabulary terms from its own feature text for each specialist view. Vocabulary selection is governed by document-frequency constraints and a maximum feature count. This produces compact local feature spaces for training.

The fourth transformation is privacy-preserving vocabulary alignment. Local vocabulary terms are converted into deterministic cryptographic tags under label and subcategory namespaces. A shared global vocabulary is then formed from these tags. The alignment mechanism allows organizations to map local feature vectors into a common parameter space without exposing plaintext vocabulary terms in the global vocabulary.

The fifth transformation is feature-matrix generation. Each organization's rows are encoded into numeric matrices using the local vocabulary associated with the relevant specialist. The experiment uses term-frequency-style representations for inference and later training rounds, and a TF-IDF-style representation for the initial training round.

Prediction generation produces structured event-level outputs. Each output contains row identity, predicted class, class probabilities, risk status, specialist scores, and contribution-oriented evidence. These outputs support both performance analysis and downstream IOC generation.

IOC generation is a separate data-production process. It takes the subset of rows designated as high risk, returns to the original event content, and searches for validated observables. Supported observable categories are URLs, IPv4 addresses, domains, MD5 hashes, SHA1 hashes, and SHA256 hashes.

Indicator extraction includes normalization, validation, and filtering. URL hosts are handled separately from URL paths. IPv4 and domain candidates must satisfy syntactic validity checks. Hash candidates are filtered to avoid treating identifier-like hexadecimal values as file hashes unless the surrounding context indicates that the value is actually a hash.

The result of IOC generation is a structured set of indicator records and a STIX-style bundle. These artifacts connect the model-selected high-risk events to concrete observables that can be reviewed, shared, or operationalized.

## Model training

The learning problem is formulated as a federated, hierarchical, multi-class classification task. The hierarchy is constructed by associating each class label with one or more specialist evidence categories. Each specialist is trained as a binary one-vs-rest classifier for a particular label and evidence category.

This design decomposes the global classification task into interpretable subproblems. A network specialist evaluates whether network evidence supports a given label. A system specialist evaluates host and process evidence. Identity, cloud, LLM/tool, and cross-category specialists contribute additional evidence where applicable. Final prediction is obtained by combining these specialist perspectives.

Local training occurs independently within each organization. For a given specialist, rows belonging to the specialist's target label are treated as positive examples, and all other rows are treated as negative examples. This one-vs-rest formulation is repeated across labels and subcategories.

Because one-vs-rest specialists may encounter class imbalance, the training procedure supports balanced weighting of positive and negative examples. This reduces the extent to which a specialist is dominated by the more frequent negative class.

After local training, parameter updates are aggregated across organizations. Aggregation may weight organizations by local sample count or treat organizations uniformly. The aggregated parameters define the updated global specialist model for the next round.

The ensemble prediction mechanism operates at the logit level. Each specialist produces a score for its label-specific evidence view. For each label, the relevant specialist scores are weighted and summed with a label bias. The fused label scores are then normalized into class probabilities. The predicted class is the label with the highest probability.

This training design has several academic motivations:

- It separates heterogeneous security evidence into semantically meaningful views.
- It allows organizations to contribute to a shared model without centralizing raw logs.
- It aligns vocabularies through deterministic tags rather than plaintext feature sharing.
- It supports class-specific specialization rather than assuming that every label should use the same evidence structure.
- It produces explanation-oriented contribution data that can support analyst review and IOC generation.

The trained artifact is therefore not a single flat classifier. It is a collection of label- and subcategory-specific specialists plus fusion metadata that defines how those specialists jointly determine final risk predictions.
