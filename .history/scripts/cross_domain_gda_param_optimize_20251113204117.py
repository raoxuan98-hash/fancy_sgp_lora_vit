#!/usr/bin/env python3
"""Optimise GDA regularisation parameters with Optuna for cross-domain datasets.

This utility follows the requested evaluation protocol:

* Datasets: resisc45, imagenet-r, caltech-101, dtd (concatenated)
* Model: vit-b-p16 (pre-trained)
* No training - direct optimization on pretrained model features
* QDA parameters constrained to sum to 1

For concatenated datasets, we:
    1. Load a pretrained model without training
    2. Extract features from all concatenated datasets
    3. Use Optuna to search for best LDA/QDA regularisation strengths.
    4. Save optimisation summaries (best parameters, trial history, metrics).

The script requires access to datasets configured in ``utils/data1.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import optuna
import torch

# Add current directory to Python path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder
from trainer import build_log_dirs, train_single_run


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _import_default_args() -> Dict:
    """Load default arguments from ``main.build_parser`` without side effects."""

    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    import main as main_module  # pylint: disable=import-outside-toplevel

    parser = main_module.build_parser()
    ns = parser.parse_args([])
    ns.smart_defaults = False
    ns.test = False

    base_args = vars(ns).copy()

    if prev_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda

    return base_args


def _prepare_args(template: Mapping, seed: int) -> Dict:
    args = copy.deepcopy(dict(template))
    args.update(
        {
            "dataset": "cross_domain_concatenated",
            "cross_domain": True,
            "cross_domain_datasets": ['resisc45', 'imagenet-r', 'caltech-101', 'dtd'],
            "init_cls": 0,  # Not used in cross-domain
            "increment": 0,  # Not used in cross-domain
            "seed": seed,
            "seed_list": [seed],
            "run_id": 0,
            "vit_type": "vit-b-p16",  # Use vit-b-p16 as requested
            "iterations": 0,  # No training iterations
            "num_shots": 0,  # No few-shot sampling
            "eval_only": True,  # Evaluation only mode
        }
    )
    return args


def _configure_logging(log_dir: str) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=os.path.join(log_dir, "record.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _select_reference_stats(variants: Mapping[str, Dict]) -> Tuple[str, Dict]:
    for variant, stats in variants.items():
        if stats:
            return variant, stats
    raise RuntimeError("No valid Gaussian statistics found in variants.")


def _collect_test_features(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone features and labels for the full test loader once."""

    model.network.eval()
    features = []
    targets = []
    device = model._device  # pylint: disable=protected-access

    with torch.no_grad():
        for batch in model.test_loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            feats = model.network.forward_features(inputs).cpu()
            features.append(feats)
            targets.append(labels.cpu())

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _evaluate_classifier(classifier, features: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy (%) of a classifier on precomputed features."""

    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    if features.device != classifier_device:
        features = features.to(classifier_device)

    with torch.no_grad():
        logits = classifier(features)
        preds = logits.argmax(dim=1).cpu()

    accuracy = (preds == targets).float().mean().item() * 100.0
    return float(round(accuracy, 4))


def _optimise_classifier(
    study_name: str,
    objective_fn,
    n_trials: int,
    dataset_dir: Path,
    seed: int,
) -> Tuple[optuna.Study, Dict]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    trials_payload = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "metrics": t.user_attrs.get("metrics", {}),
        }
        for t in study.trials
    ]

    with open(dataset_dir / f"{study_name}_trials.json", "w", encoding="utf-8") as f:
        json.dump(trials_payload, f, ensure_ascii=False, indent=2)

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs.get("metrics", {}),
    }

    return study, best_payload


def optimise_concatenated_datasets(
    base_args: Mapping,
    seed: int,
    n_trials: int,
    output_dir: Path,
    lda_range: Tuple[float, float],
    qda_alpha1_range: Tuple[float, float],
    qda_alpha2_range: Tuple[float, float],
    qda_alpha3_range: Tuple[float, float],
    eval_device: torch.device,
    cache_features_on_device: bool,
) -> Dict:
    """Optimize GDA parameters on concatenated cross-domain datasets without training."""
    logging.info("===== Processing concatenated cross-domain datasets (no training) =====")
    
    # Use evaluation-only mode to load pretrained model
    args = _prepare_args(base_args, seed)
    
    _, log_dir = build_log_dirs(args)
    args["log_path"] = log_dir
    _configure_logging(log_dir)

    # Load pretrained model without training
    results, model = train_single_run(args, return_model=True)
    logging.info("Model loaded. Results: %s", results)

    variants = model.drift_compensator.variants
    device = model._device  # pylint: disable=protected-access
    selected_variant, reference_stats = _select_reference_stats(variants)
    logging.info("Using %s statistics for optimisation.", selected_variant)
    features_cpu, targets = _collect_test_features(model)
    device_for_classifiers = str(eval_device)

    if cache_features_on_device and eval_device.type != "cpu":
        features_for_objective = features_cpu.to(eval_device)
    else:
        features_for_objective = features_cpu

    dataset_dir = output_dir / "concatenated"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    def lda_objective(trial: optuna.Trial) -> float:
        reg_alpha = trial.suggest_float("reg_alpha", lda_range[0], lda_range[1])
        builder = LDAClassifierBuilder(reg_alpha=reg_alpha, device=device_for_classifiers)
        classifier = builder.build(reference_stats)
        accuracy = _evaluate_classifier(classifier, features_for_objective, targets)
        metrics = {selected_variant: accuracy}
        trial.set_user_attr("metrics", metrics)
        del classifier
        return accuracy

    def qda_objective(trial: optuna.Trial) -> float:
        # QDA parameters with constraint that they sum to 1
        alpha1 = trial.suggest_float("alpha1", qda_alpha1_range[0], qda_alpha1_range[1])
        alpha2 = trial.suggest_float("alpha2", qda_alpha2_range[0], qda_alpha2_range[1])
        alpha3 = trial.suggest_float("alpha3", qda_alpha3_range[0], qda_alpha3_range[1])
        
        # Normalize to ensure sum = 1
        total = alpha1 + alpha2 + alpha3
        reg_alpha1 = alpha1 / total
        reg_alpha2 = alpha2 / total
        reg_alpha3 = alpha3 / total
        
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=reg_alpha1,
            qda_reg_alpha2=reg_alpha2,
            qda_reg_alpha3=reg_alpha3,
            device=device_for_classifiers,
        )
        classifier = builder.build(reference_stats)
        accuracy = _evaluate_classifier(classifier, features_for_objective, targets)
        metrics = {selected_variant: accuracy}
        trial.set_user_attr("metrics", metrics)
        del classifier
        return accuracy

    _, lda_summary = _optimise_classifier("lda", lda_objective, n_trials, dataset_dir, seed)
    _, qda_summary = _optimise_classifier("qda", qda_objective, n_trials, dataset_dir, seed + 1)

    dataset_summary = {
        "dataset": "cross_domain_concatenated",
        "datasets_used": ["resisc45", "imagenet-r", "caltech-101", "dtd"],
        "seed": seed,
        "reference_variant": selected_variant,
        "lda": lda_summary,
        "qda": qda_summary,
    }

    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

    # Explicitly release resources tied to model and cached tensors
    del model
    if cache_features_on_device and eval_device.type != "cpu":
        del features_for_objective
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_optutna/cross_domain_gda_param_optuna_vit_b16"),
        help="Directory to store optimisation artefacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="Random seed used for model loading.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for each classifier type.",
    )
    parser.add_argument(
        "--lda-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for LDA reg_alpha.",
    )
    parser.add_argument(
        "--qda-alpha1-range",
        type=float,
        nargs=2,
        default=(0.01, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for QDA reg_alpha1.",
    )
    parser.add_argument(
        "--qda-alpha2-range",
        type=float,
        nargs=2,
        default=(0.01, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for QDA reg_alpha2.",
    )
    parser.add_argument(
        "--qda-alpha3-range",
        type=float,
        nargs=2,
        default=(0.01, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for QDA reg_alpha3.",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default="cpu",
        help="Device used for classifier evaluation (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--cache-features-on-device",
        action="store_true",
        help="Keep extracted features on evaluation device. Disable to save GPU memory.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    base_args = _import_default_args()

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    eval_device = torch.device(cli_args.eval_device)

    # Optimize on concatenated datasets as specified in requirements
    logging.info("Starting optimization on concatenated cross-domain datasets (no training)...")
    concatenated_summary = optimise_concatenated_datasets(
        base_args=base_args,
        seed=cli_args.seed,
        n_trials=cli_args.n_trials,
        output_dir=cli_args.output_dir,
        lda_range=tuple(cli_args.lda_range),
        qda_alpha1_range=tuple(cli_args.qda_alpha1_range),
        qda_alpha2_range=tuple(cli_args.qda_alpha2_range),
        qda_alpha3_range=tuple(cli_args.qda_alpha3_range),
        eval_device=eval_device,
        cache_features_on_device=cli_args.cache_features_on_device,
    )
    all_summaries["concatenated"] = concatenated_summary

    with open(cli_args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()