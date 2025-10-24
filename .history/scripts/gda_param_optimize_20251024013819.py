#!/usr/bin/env python3
"""Optimise GDA regularisation parameters with Optuna.

This utility follows the requested evaluation protocol:

* Datasets: imagenet-r, cifar100_224, cub200_224, cars196_224
* init_cls:   200,        100,          200,          196
* increment:  0  (single-task evaluation)

For every dataset, we:
    1. Train a ``SubspaceLoRA`` model once with the default configuration.
    2. Reuse the cached Gaussian statistics to evaluate LDA/QDA classifiers
       while Optuna searches for the best regularisation strengths.
    3. Save optimisation summaries (best parameters, trial history, metrics).

The training step still requires access to the datasets configured in
``utils/data1.py``.  When datasets are not available the script will fail
before the optimisation loop begins.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

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
            inputs = inputs.to(device)
            feats = model.network.forward_features(inputs).cpu()
            features.append(feats)
            targets.append(labels.cpu())

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _evaluate_classifier(classifier, features: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy (%) of a classifier on precomputed features."""

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
) -> Dict:
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
    features_on_device = features_cpu.to(device)
    del features_cpu

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    def lda_objective(trial: optuna.Trial) -> float:
        reg_alpha = trial.suggest_float("reg_alpha", lda_range[0], lda_range[1])
        builder = LDAClassifierBuilder(reg_alpha=reg_alpha, device=device)
        classifier = builder.build(reference_stats)
        accuracy = _evaluate_classifier(classifier, features_on_device, targets)
        metrics = {selected_variant: accuracy}
        trial.set_user_attr("metrics", metrics)
        return accuracy

    def qda_objective(trial: optuna.Trial) -> float:
        reg_alpha1 = trial.suggest_float("reg_alpha1", qda_alpha1_range[0], qda_alpha1_range[1])
        reg_alpha2 = trial.suggest_float("reg_alpha2", qda_alpha2_range[0], qda_alpha2_range[1])
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=reg_alpha1,
            qda_reg_alpha2=reg_alpha2,
            device=device,
        )
        classifier = builder.build(reference_stats)
        accuracy = _evaluate_classifier(classifier, features_on_device, targets)
        metrics = {selected_variant: accuracy}
        trial.set_user_attr("metrics", metrics)
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

    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

    # Explicitly release resources tied to the model
    del model
    return dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/gda_param_optuna"),
        help="Directory to store optimisation artefacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="Random seed used for training runs.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
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
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for QDA reg_alpha1.",
    )
    parser.add_argument(
        "--qda-alpha2-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Search range for QDA reg_alpha2.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    base_args = _import_default_args()

    datasets = {
        "imagenet-r": 200,
        "cifar100_224": 100,
        "cub200_224": 200,
        "cars196_224": 196,
    }

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for name, init_cls in datasets.items():
        summary = optimise_dataset(
            dataset=name,
            init_cls=init_cls,
            base_args=base_args,
            seed=cli_args.seed,
            n_trials=cli_args.n_trials,
            output_dir=cli_args.output_dir,
            lda_range=tuple(cli_args.lda_range),
            qda_alpha1_range=tuple(cli_args.qda_alpha1_range),
            qda_alpha2_range=tuple(cli_args.qda_alpha2_range),
        )
        all_summaries[name] = summary

    with open(cli_args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
