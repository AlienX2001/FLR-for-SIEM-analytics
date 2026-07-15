from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from federated_lr_pipeline.config import PipelineConfig, parse_args
from federated_lr_pipeline.data import OrgDataset, load_all_orgs
from federated_lr_pipeline.ensemble import ManualLogitFusion
from federated_lr_pipeline.prf import derive_prf_key
from federated_lr_pipeline.specialized_models import (
    active_subcategories_for_hierarchy,
    build_hierarchical_config,
    build_subcategory_token_counters,
    build_subcategory_texts,
    evaluate_hierarchical_ensemble,
    generate_hierarchical_predictions,
    initialize_all_specialists,
    save_hierarchical_artifacts,
    train_specialist_round,
    write_hierarchical_inference_outputs,
)
from federated_lr_pipeline.utils import ensure_dir, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _configure_parallel_environment(num_workers: int) -> None:
    if num_workers <= 1:
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    LOGGER.info(
        "Using %s local training worker threads with BLAS thread limits set when unset",
        num_workers,
    )


@dataclass(frozen=True)
class OrgTrainTestSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _fit_label_encoder(org_datasets: list[OrgDataset]) -> tuple[LabelEncoder, list[np.ndarray]]:
    encoder = LabelEncoder()
    all_labels = [label for dataset in org_datasets for label in dataset.labels]
    encoder.fit(all_labels)
    encoded_by_org = [
        encoder.transform(dataset.labels).astype(int) for dataset in org_datasets
    ]
    return encoder, encoded_by_org


def _create_stratified_splits(
    org_datasets: list[OrgDataset],
    encoded_labels_by_org: list[np.ndarray],
    *,
    test_size: float,
    seed: int,
) -> list[OrgTrainTestSplit]:
    splits: list[OrgTrainTestSplit] = []
    for dataset, labels in zip(org_datasets, encoded_labels_by_org):
        n_samples = len(labels)
        if n_samples < 2:
            raise ValueError(
                f"Organization {dataset.org_index} needs at least 2 rows for a train/test split"
            )
        class_counts = pd.Series(labels).value_counts()
        if class_counts.min() < 2:
            raise ValueError(
                f"Organization {dataset.org_index} cannot use a stratified split because "
                f"at least one local class has fewer than 2 rows: {class_counts.to_dict()}"
            )
        n_classes_local = len(class_counts)
        n_test = int(np.ceil(test_size * n_samples))
        n_train = n_samples - n_test
        if n_test < n_classes_local or n_train < n_classes_local:
            raise ValueError(
                f"Organization {dataset.org_index} test_size={test_size} leaves too few "
                f"train/test rows for stratification over {n_classes_local} local classes"
            )
        row_positions = np.arange(n_samples, dtype=int)
        train_indices, test_indices = train_test_split(
            row_positions,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        splits.append(
            OrgTrainTestSplit(
                train_indices=np.asarray(train_indices, dtype=int),
                test_indices=np.asarray(test_indices, dtype=int),
            )
        )
    return splits


def run_pipeline(config: PipelineConfig) -> None:
    if config.testing:
        from federated_lr_pipeline.testing import run_testing_mode

        run_testing_mode(config)
        return

    setup_logging()
    _configure_parallel_environment(config.num_workers)
    output_dir = ensure_dir(config.output_dir)
    write_json(output_dir / "run_config.json", config.to_json_dict())

    LOGGER.info("Loading row-aligned organization data")
    org_datasets = load_all_orgs(
        config.org_data,
        config.org_groundtruth,
        text_column=config.text_column,
        text_columns=config.text_columns,
        label_column=config.label_column,
    )

    LOGGER.info("Encoding labels")
    label_encoder, encoded_labels_by_org = _fit_label_encoder(org_datasets)
    label_classes = [str(label) for label in label_encoder.classes_.tolist()]
    write_json(output_dir / "label_encoder_classes.json", label_classes)

    LOGGER.info("Creating stratified train/test splits")
    splits = _create_stratified_splits(
        org_datasets,
        encoded_labels_by_org,
        test_size=config.test_size,
        seed=config.seed,
    )

    LOGGER.info("Building label-conditioned hierarchy")
    hierarchy = build_hierarchical_config(
        label_classes,
        config.hierarchical_config,
        fusion_mode=config.fusion_mode,
    )
    write_json(
        output_dir / "hierarchical_config.json",
        {
            "labels": {
                label: {
                    "subcategories": branch.subcategories,
                    "weights": branch.weights,
                }
                for label, branch in hierarchy.branches.items()
            },
            "ensemble": {
                "fusion": hierarchy.fusion,
                "fusion_mode": hierarchy.fusion_mode,
            },
        },
    )

    LOGGER.info("Building subcategory field-aware inputs")
    active_subcategories = active_subcategories_for_hierarchy(hierarchy)
    texts_by_subcategory, missing_by_subcategory = build_subcategory_texts(
        org_datasets,
        subcategories=active_subcategories,
    )
    token_counters_by_subcategory = build_subcategory_token_counters(texts_by_subcategory)

    LOGGER.info("Initializing label/subcategory vocabularies and specialists")
    specialists = initialize_all_specialists(
        hierarchy=hierarchy,
        org_datasets=org_datasets,
        texts_by_subcategory=texts_by_subcategory,
        token_counters_by_subcategory=token_counters_by_subcategory,
        missing_by_subcategory=missing_by_subcategory,
        splits=splits,
        config=config,
        prf_key=derive_prf_key(config.seed),
    )
    fusion = ManualLogitFusion(
        labels=hierarchy.labels,
        subcategories_by_label={
            label: branch.subcategories for label, branch in hierarchy.branches.items()
        },
        weights_by_label={
            label: branch.weights for label, branch in hierarchy.branches.items()
        },
    )

    metrics_by_round: list[dict[str, object]] = []
    total_rounds = config.federation_iterations
    label_to_index = {label: index for index, label in enumerate(label_classes)}

    for round_index in range(total_rounds):
        mode = "tfidf" if round_index == 0 else "tf"
        train_feature_matrix_cache = {}
        LOGGER.info(
            "Starting hierarchical federation round %s/%s using %s features",
            round_index + 1,
            total_rounds,
            mode,
        )
        specialist_metrics: dict[str, dict[str, object]] = {}
        for label in hierarchy.labels:
            specialist_metrics[label] = {}
            for subcategory, state in specialists[label].items():
                specialist_metrics[label][subcategory] = train_specialist_round(
                    state=state,
                    org_datasets=org_datasets,
                    encoded_labels_by_org=encoded_labels_by_org,
                    label_index=label_to_index[label],
                    splits=splits,
                    mode=mode,
                    round_index=round_index,
                    total_rounds=total_rounds,
                    config=config,
                    feature_matrix_cache=train_feature_matrix_cache,
                )

        test_feature_matrix_cache = {}
        ensemble_metrics = evaluate_hierarchical_ensemble(
            specialists=specialists,
            fusion=fusion,
            encoded_labels_by_org=encoded_labels_by_org,
            splits=splits,
            class_names=label_classes,
            feature_matrix_cache=test_feature_matrix_cache,
        )
        metrics_by_round.append(
            {
                "round": round_index,
                "round_number": round_index + 1,
                "feature_mode": mode,
                "specialists": specialist_metrics,
                "ensemble_metrics": ensemble_metrics,
            }
        )
        LOGGER.info(
            "Completed hierarchical federation round %s/%s "
            "(ensemble_test_accuracy=%.4f)",
            round_index + 1,
            total_rounds,
            ensemble_metrics["test_accuracy"],
        )

    LOGGER.info("Saving final hierarchical model artifacts")
    save_hierarchical_artifacts(
        output_dir=output_dir,
        specialists=specialists,
        debug_plaintext_vocab=config.debug_plaintext_vocab,
    )
    fusion.save(output_dir / "manual_logit_fusion.json")
    write_json(
        output_dir / "training_metrics.json",
        {
            "hierarchy": {
                label: {
                    "subcategories": branch.subcategories,
                    "weights": branch.weights,
                }
                for label, branch in hierarchy.branches.items()
            },
            "split": {
                "test_size": config.test_size,
                "per_org": [
                    {
                        "org_index": dataset.org_index,
                        "num_train_examples": len(split.train_indices),
                        "num_test_examples": len(split.test_indices),
                    }
                    for dataset, split in zip(org_datasets, splits)
                ],
            },
            "aggregation": {"weighting": config.aggregation_weighting},
            "class_weight": {"mode": config.class_weight},
            "vocabulary_source": config.vocabulary_source,
            "rounds": metrics_by_round,
        },
    )

    LOGGER.info("Running hierarchical logit-fusion inference")
    prediction_records = generate_hierarchical_predictions(
        org_datasets=org_datasets,
        specialists=specialists,
        fusion=fusion,
        label_classes=label_classes,
        risk_threshold=config.risk_threshold,
        debug_plaintext_vocab=config.debug_plaintext_vocab,
    )
    write_hierarchical_inference_outputs(prediction_records, output_dir)
    LOGGER.info("Wrote outputs to %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    if config.testing:
        from federated_lr_pipeline.testing import run_testing_mode

        run_testing_mode(config)
        return
    run_pipeline(config)


if __name__ == "__main__":
    main()
