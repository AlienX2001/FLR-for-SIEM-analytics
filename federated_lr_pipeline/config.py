from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    org_data: list[Path]
    org_groundtruth: list[Path]
    num_features: int
    federation_iterations: int
    min_df: int = 2
    max_df: float = 0.95
    output_dir: Path = Path("outputs/run_001")
    text_column: str | None = None
    text_columns: list[str] | None = None
    label_column: str | None = None
    hierarchical_config: Path | None = None
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.05
    local_epochs: int = 1
    regularization: float = 1e-4
    risk_threshold: float = 0.75
    local_progress_interval: int = 0
    num_workers: int = 1
    test_size: float = 0.2
    class_weight: str = "balanced"
    vocabulary_source: str = "train"
    aggregation_weighting: str = "sample_size"
    fusion_mode: str = "manual"
    use_global_model: bool = False
    debug_plaintext_vocab: bool = False
    testing: bool = False
    model_artifact_dir: Path | None = None
    network_weights: Path | None = None
    network_bias: Path | None = None
    system_weights: Path | None = None
    system_bias: Path | None = None
    inter_category_weights: Path | None = None
    inter_category_bias: Path | None = None
    network_gv_tags: Path | None = None
    system_gv_tags: Path | None = None
    inter_category_gv_tags: Path | None = None
    label_encoder_classes: Path | None = None
    run_config: Path | None = None
    network_index_vectors: list[Path] | None = None
    system_index_vectors: list[Path] | None = None
    inter_category_index_vectors: list[Path] | None = None
    network_lv_tokens: list[Path] | None = None
    system_lv_tokens: list[Path] | None = None
    inter_category_lv_tokens: list[Path] | None = None
    hierarchical_model_manifest: Path | None = None
    manual_logit_fusion: Path | None = None
    ensemble_method: str | None = None
    network_logit_weight: float = 1.0
    system_logit_weight: float = 1.0
    inter_logit_weight: float = 1.0
    testing_ignored_parameters: list[str] = field(default_factory=list)
    testing_override_parameters: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["org_data"] = [str(path) for path in self.org_data]
        payload["org_groundtruth"] = [str(path) for path in self.org_groundtruth]
        payload["output_dir"] = str(self.output_dir)
        if self.model_artifact_dir is not None:
            payload["model_artifact_dir"] = str(self.model_artifact_dir)
        payload["effective_label_column_default"] = self.label_column or "label"
        return payload


def _prompt_path_list(prompt: str, parser: argparse.ArgumentParser) -> list[Path]:
    if not sys.stdin.isatty():
        parser.error(f"{prompt} is required in --testing mode when stdin is not interactive")
    raw_value = input(f"{prompt}, comma-separated: ").strip()
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        parser.error(f"{prompt} cannot be empty")
    return [Path(value) for value in values]


def _prompt_path(prompt: str, parser: argparse.ArgumentParser) -> Path:
    if not sys.stdin.isatty():
        parser.error(f"{prompt} is required in --testing mode when stdin is not interactive")
    raw_value = input(f"{prompt}: ").strip()
    if not raw_value:
        parser.error(f"{prompt} cannot be empty")
    return Path(raw_value)


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    provided_flags = {
        token.split("=", maxsplit=1)[0]
        for token in raw_argv
        if token.startswith("--")
    }
    parser = argparse.ArgumentParser(
        description="Privacy-preserving federated logistic regression prototype."
    )
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--org-data", nargs="+", type=Path)
    parser.add_argument("--org-groundtruth", nargs="+", type=Path)
    parser.add_argument("--num-features", type=int)
    parser.add_argument("--federation-iterations", type=int)
    parser.add_argument("--min-df", default=2, type=int)
    parser.add_argument("--max-df", default=0.95, type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text-column", default=None)
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=None,
        help="One or more log CSV columns to concatenate as text.",
    )
    parser.add_argument("--label-column", default=None)
    parser.add_argument(
        "--hierarchical-config",
        default=None,
        type=Path,
        help="Optional JSON file defining label subcategories and manual fusion weights.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--local-epochs", default=1, type=int)
    parser.add_argument("--regularization", default=1e-4, type=float)
    parser.add_argument("--risk-threshold", default=0.75, type=float)
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Held-out test fraction per organization for stratified evaluation.",
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Class weighting for local LR training. Default: balanced.",
    )
    parser.add_argument(
        "--vocabulary-source",
        choices=("train", "all"),
        default="train",
        help=(
            "Rows used for local vocabulary generation. Default: train, which avoids "
            "using held-out test rows for feature selection."
        ),
    )
    parser.add_argument(
        "--aggregation-weighting",
        choices=("sample_size", "uniform"),
        default="sample_size",
        help="Organization weighting used when aggregating specialist parameters.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("manual", "meta"),
        default="manual",
        help="Manual logit fusion is implemented; meta is reserved for later.",
    )
    parser.add_argument(
        "--use-global-model",
        action="store_true",
        help="Reserved compatibility flag. The default hierarchical ensemble does not use it.",
    )
    parser.add_argument(
        "--debug-plaintext-vocab",
        action="store_true",
        help="Include plaintext feature tokens in inference explanations.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        default=None,
        type=Path,
        help="Directory containing trained artifacts for --testing mode.",
    )
    parser.add_argument("--network-weights", default=None, type=Path)
    parser.add_argument("--network-bias", default=None, type=Path)
    parser.add_argument("--system-weights", default=None, type=Path)
    parser.add_argument("--system-bias", default=None, type=Path)
    parser.add_argument("--inter-category-weights", default=None, type=Path)
    parser.add_argument("--inter-category-bias", default=None, type=Path)
    parser.add_argument("--network-gv-tags", default=None, type=Path)
    parser.add_argument("--system-gv-tags", default=None, type=Path)
    parser.add_argument("--inter-category-gv-tags", default=None, type=Path)
    parser.add_argument("--label-encoder-classes", default=None, type=Path)
    parser.add_argument("--run-config", default=None, type=Path)
    parser.add_argument("--network-index-vectors", nargs="+", default=None, type=Path)
    parser.add_argument("--system-index-vectors", nargs="+", default=None, type=Path)
    parser.add_argument("--inter-category-index-vectors", nargs="+", default=None, type=Path)
    parser.add_argument("--network-lv-tokens", nargs="+", default=None, type=Path)
    parser.add_argument("--system-lv-tokens", nargs="+", default=None, type=Path)
    parser.add_argument("--inter-category-lv-tokens", nargs="+", default=None, type=Path)
    parser.add_argument("--hierarchical-model-manifest", default=None, type=Path)
    parser.add_argument("--manual-logit-fusion", default=None, type=Path)
    parser.add_argument(
        "--ensemble-method",
        choices=("average_logits", "weighted_average_logits"),
        default=None,
        help="Testing-only override for subcategory logit fusion.",
    )
    parser.add_argument("--network-logit-weight", default=1.0, type=float)
    parser.add_argument("--system-logit-weight", default=1.0, type=float)
    parser.add_argument("--inter-logit-weight", default=1.0, type=float)
    parser.add_argument(
        "--local-progress-interval",
        "--log-local-iterations",
        dest="local_progress_interval",
        default=0,
        type=int,
        help=(
            "Log every N local feature rows and mini-batch updates. "
            "Use 0 to disable per-N local progress logs."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Parallel worker threads for independent local organization training. Default: 1.",
    )
    args = parser.parse_args(argv)

    if args.testing:
        if args.org_data is None:
            args.org_data = _prompt_path_list("Enter raw log CSV paths for each organization", parser)
        if args.org_groundtruth is None:
            args.org_groundtruth = _prompt_path_list(
                "Enter row-aligned groundtruth CSV paths for each organization",
                parser,
            )
        if args.model_artifact_dir is None and args.hierarchical_model_manifest is None:
            args.model_artifact_dir = _prompt_path("Enter pretrained artifact directory", parser)
        if args.output_dir is None:
            args.output_dir = _prompt_path("Enter testing output directory", parser)
    else:
        if args.org_data is None:
            parser.error("--org-data is required")
        if args.org_groundtruth is None:
            parser.error("--org-groundtruth is required")
        if args.num_features is None:
            parser.error("--num-features is required")
        if args.federation_iterations is None:
            parser.error("--federation-iterations is required")
        if args.output_dir is None:
            parser.error("--output-dir is required")

    if len(args.org_data) != len(args.org_groundtruth):
        parser.error("--org-data and --org-groundtruth must contain the same number of files")
    if args.text_column is not None and args.text_columns is not None:
        parser.error("Use either --text-column or --text-columns, not both")
    if not args.testing and args.num_features is not None and args.num_features <= 0:
        parser.error("--num-features must be positive")
    if not args.testing and args.federation_iterations is not None and args.federation_iterations < 0:
        parser.error("--federation-iterations must be non-negative")
    if not args.testing and args.min_df < 1:
        parser.error("--min-df must be at least 1")
    if not args.testing and args.max_df <= 0:
        parser.error("--max-df must be positive")
    if not args.testing and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.testing and args.local_epochs < 0:
        parser.error("--local-epochs must be non-negative")
    if not 0 <= args.risk_threshold <= 1:
        parser.error("--risk-threshold must be in [0, 1]")
    if not args.testing and args.local_progress_interval < 0:
        parser.error("--local-progress-interval must be non-negative")
    if not args.testing and args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if not args.testing and not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    if not args.testing and args.fusion_mode == "meta":
        parser.error("--fusion-mode meta is a placeholder and is not implemented yet")
    if args.use_global_model:
        parser.error("--use-global-model is reserved for compatibility and is not implemented")
    if args.testing and args.model_artifact_dir is None and args.hierarchical_model_manifest is None:
        parser.error(
            "--testing requires --model-artifact-dir or --hierarchical-model-manifest"
        )

    training_flags = {
        "--num-features",
        "--federation-iterations",
        "--min-df",
        "--max-df",
        "--learning-rate",
        "--local-epochs",
        "--regularization",
        "--batch-size",
        "--num-workers",
    }
    testing_override_flags = {
        "--risk-threshold",
        "--ensemble-method",
        "--network-logit-weight",
        "--system-logit-weight",
        "--inter-logit-weight",
        "--debug-plaintext-vocab",
    }
    ignored_testing_parameters = sorted(training_flags & provided_flags) if args.testing else []
    testing_override_parameters = sorted(testing_override_flags & provided_flags)

    return PipelineConfig(
        org_data=args.org_data,
        org_groundtruth=args.org_groundtruth,
        num_features=args.num_features if args.num_features is not None else 0,
        federation_iterations=(
            args.federation_iterations if args.federation_iterations is not None else 0
        ),
        min_df=args.min_df,
        max_df=args.max_df,
        output_dir=args.output_dir,
        text_column=args.text_column,
        text_columns=args.text_columns,
        label_column=args.label_column,
        hierarchical_config=args.hierarchical_config,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        regularization=args.regularization,
        risk_threshold=args.risk_threshold,
        local_progress_interval=args.local_progress_interval,
        num_workers=args.num_workers,
        test_size=args.test_size,
        class_weight=args.class_weight,
        vocabulary_source=args.vocabulary_source,
        aggregation_weighting=args.aggregation_weighting,
        fusion_mode=args.fusion_mode,
        use_global_model=args.use_global_model,
        debug_plaintext_vocab=args.debug_plaintext_vocab,
        testing=args.testing,
        model_artifact_dir=args.model_artifact_dir,
        network_weights=args.network_weights,
        network_bias=args.network_bias,
        system_weights=args.system_weights,
        system_bias=args.system_bias,
        inter_category_weights=args.inter_category_weights,
        inter_category_bias=args.inter_category_bias,
        network_gv_tags=args.network_gv_tags,
        system_gv_tags=args.system_gv_tags,
        inter_category_gv_tags=args.inter_category_gv_tags,
        label_encoder_classes=args.label_encoder_classes,
        run_config=args.run_config,
        network_index_vectors=args.network_index_vectors,
        system_index_vectors=args.system_index_vectors,
        inter_category_index_vectors=args.inter_category_index_vectors,
        network_lv_tokens=args.network_lv_tokens,
        system_lv_tokens=args.system_lv_tokens,
        inter_category_lv_tokens=args.inter_category_lv_tokens,
        hierarchical_model_manifest=args.hierarchical_model_manifest,
        manual_logit_fusion=args.manual_logit_fusion,
        ensemble_method=args.ensemble_method,
        network_logit_weight=args.network_logit_weight,
        system_logit_weight=args.system_logit_weight,
        inter_logit_weight=args.inter_logit_weight,
        testing_ignored_parameters=ignored_testing_parameters,
        testing_override_parameters=testing_override_parameters,
    )
