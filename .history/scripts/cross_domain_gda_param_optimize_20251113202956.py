#!/usr/bin/env python3
"""Optimise GDA regularisation parameters with Optuna for Cross-Domain datasets.

This utility follows the requested evaluation protocol for cross-domain experiments:

* Datasets: resisc45, imagenet-r, caltech-101, dtd
* Model: vit-b-p16 (pre-trained)
* QDA constraint: reg_alpha1 + reg_alpha2 + reg_alpha3 = 1.0

For every dataset, we:
    1. Train a ``SubspaceLoRA`` model once with the default configuration.
    2. Reuse the cached Gaussian statistics to evaluate LDA/QDA classifiers
       while Optuna searches for the best regularisation strengths.
    3. Save optimisation summaries (best parameters, trial history, metrics).

The training step still requires access to the datasets configured in
``utils/data1.py``.  When datasets are not available the script will fail
before the optimisation loop begins.

Command-line options allow:

* Joint optimisation across all datasets for LDA/QDA regularisation strengths.
* Selecting the device used for classifier evaluation while optionally avoiding
  caching features on the GPU to reduce memory pressure.
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


def _prepare_args(template: Mapping, dataset: str, init_cls: int, seed: int) -> Dict:
    args = copy.deepcopy(dict(template))
    args.update(
        {
            "dataset": dataset,
            "init_cls": init_cls,
            "increment": 0,
            "seed": seed,
            "seed_list": [seed],
            "run_id": 0,
            "cross_domain": True,
            "cross_domain_datasets": ["resisc45", "imagenet-r", "caltech-101", "dtd"],
            "vit_type": "vit-b-p16",
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


def optimise_dataset(
    dataset: str,
    init_cls: int,
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
    retain_features: bool,
) -> Tuple[Dict, Optional[Dict[str, Any]]]:
    logging.info("===== Processing %s (init_cls=%d) =====", dataset, init_cls)
    args = _prepare_args(base_args, dataset, init_cls, seed)

    _, log_dir = build_log_dirs(args)
    args["log_path"] = log_dir
    _configure_logging(log_dir)

    results, model = train_single_run(args, return_model=True)
    logging.info("Training finished. Results: %s", results)

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

    if not retain_features:
        del features_cpu

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "training_results.json", "w", encoding="utf-8") as f:
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
        # QDA constraint: reg_alpha1 + reg_alpha2 + reg_alpha3 = 1.0
        # We'll use two parameters and derive the third
        reg_alpha1 = trial.suggest_float("reg_alpha1", qda_alpha1_range[0], qda_alpha1_range[1])
        reg_alpha2 = trial.suggest_float("reg_alpha2", qda_alpha2_range[0], qda_alpha2_range[1])
        
        # Ensure the sum doesn't exceed 1.0
        if reg_alpha1 + reg_alpha2 >= 1.0:
            return 0.0  # Penalize invalid combinations
        
        reg_alpha3 = 1.0 - reg_alpha1 - reg_alpha2
        
        # Check if reg_alpha3 is within valid range
        if reg_alpha3 < qda_alpha3_range[0] or reg_alpha3 > qda_alpha3_range[1]:
            return 0.0  # Penalize invalid combinations
        
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
        "dataset": dataset,
        "init_cls": init_cls,
        "seed": seed,
        "reference_variant": selected_variant,
        "lda": lda_summary,
        "qda": qda_summary,
    }

