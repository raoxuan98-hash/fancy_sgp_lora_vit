#!/usr/bin/env python3
"""Optimise GDA regularisation parameters with Optuna for cross-domain datasets.

This utility follows the requested evaluation protocol:

* Datasets: resisc45, imagenet-r, caltech-101, dtd (concatenated)
* Model: vit-b-p16 (pre-trained)
* No training - direct optimization on pretrained model features
* QDA parameters constrained to sum to 1
* iterations=1 (no optimization)

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
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
from utils.inc_net import BaseNet
from compensator.gaussian_statistics import GaussianStatistics


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def set_random(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
            "iterations": 1,  # iterations=1 as requested (no optimization)
            "num_shots": 128,  # 64-shot sampling as requested
            "eval_only": True,  # Evaluation only mode
            "lora_type": "basic_lora",  # Use basic LoRA for feature extraction
            "lora_rank": 4,  # Default LoRA rank
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


def _collect_test_features(model, data_manager, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone features and labels for the full test loader once."""

    model.eval()
    features = []
    targets = []

    with torch.no_grad():
        for task_id in range(data_manager.nb_tasks):
            # Get test dataset for this task
            dataset = data_manager.get_subset(
                task=task_id, source="test", cumulative=False, mode="test"
            )
            
            # Create data loader
            loader = DataLoader(
                dataset, batch_size=16, shuffle=False,
                num_workers=4, pin_memory=True
            )
            
            # Extract features for this task with progress bar
            task_name = data_manager.dataset_names[task_id]
            pbar = tqdm(loader, desc=f"Extracting features from {task_name}", leave=False)
            for batch in pbar:
                inputs, labels, _ = batch
                inputs = inputs.to(device)
                feats = model.forward_features(inputs).cpu()
                features.append(feats)
                targets.append(labels.cpu())
            pbar.close()

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
    
    # Set random seed for reproducibility
    set_random(seed)
    
    # Prepare arguments
    args = _prepare_args(base_args, seed)
    
    # Create output directory
    dataset_dir = output_dir / "concatenated"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = dataset_dir / "optimization.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting GDA parameter optimization on concatenated datasets")
    logging.info(f"Random seed: {seed}")
    logging.info(f"Number of trials: {n_trials}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cross-domain datasets: {args['cross_domain_datasets']}")
    logging.info(f"Using model: {args['vit_type']}")
    logging.info(f"Iterations: {args['iterations']} (no optimization)")
    
    # Initialize data manager for cross-domain datasets
    data_manager = CrossDomainDataManagerCore(
        dataset_names=args['cross_domain_datasets'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        num_shots=args.get('num_shots', 0),
        num_samples_per_task_for_evaluation=args.get('num_samples_per_task_for_evaluation', 0)
    )
    
    # Load pre-trained model directly without training
    logging.info("Loading pre-trained ViT model...")
    model = BaseNet(args, pretrained=True).to(eval_device)
    model.eval()
    
    # Extract features from all datasets
    logging.info("Extracting features from concatenated datasets...")
    all_features, all_labels = _collect_test_features(model, data_manager, eval_device)
    
    # Split features for training and testing
    num_samples = len(all_features)
    indices = torch.randperm(num_samples)
    split_idx = int(0.8 * num_samples)  # 80% for training, 20% for testing
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    test_features = all_features[test_indices]
    test_labels = all_labels[test_indices]
    
    # Move features to device if needed
    if cache_features_on_device and eval_device.type != "cpu":
        train_features = train_features.to(eval_device)
        test_features = test_features.to(eval_device)
    
    device_for_classifiers = str(eval_device)
    
    # Define the objective function for QDA (only optimize QDA, not LDA)
    def qda_objective(trial: optuna.Trial) -> float:
        # For QDA, we fix the identity matrix interpolation weight to 0.1
        # and optimize the other two parameters with the constraint that they sum to 0.9
        alpha1 = trial.suggest_float("alpha1", qda_alpha1_range[0], qda_alpha1_range[1])
        alpha2 = trial.suggest_float("alpha2", qda_alpha2_range[0], qda_alpha2_range[1])
        
        # Fix identity matrix weight to 0.1 and normalize the other two to sum to 0.9
        identity_weight = 0.1
        total = alpha1 + alpha2
        reg_alpha1 = alpha1 / total * (1.0 - identity_weight)  # class covariance weight
        reg_alpha2 = alpha2 / total * (1.0 - identity_weight)  # global covariance weight
        reg_alpha3 = identity_weight  # identity matrix weight fixed at 0.1
        
        # Build classifier using the builder pattern
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=reg_alpha1,
            qda_reg_alpha2=reg_alpha2,
            qda_reg_alpha3=reg_alpha3,
            device=device_for_classifiers,
        )
        
        # Create stats_dict for classifier builder using GaussianStatistics objects
        stats_dict = {}
        unique_labels = torch.unique(train_labels)
        for label in unique_labels:
            label_mask = train_labels == label
            class_features = train_features[label_mask]
            if len(class_features) > 0:
                mean = class_features.mean(dim=0)
                cov = torch.cov(class_features.T)
                stats_dict[int(label)] = GaussianStatistics(mean, cov)
        
        classifier = builder.build(stats_dict)
        accuracy = _evaluate_classifier(classifier, test_features, test_labels)
        metrics = {"qda": accuracy}
        trial.set_user_attr("metrics", metrics)
        del classifier
        return accuracy
    
    # Run optimization (only QDA, not LDA)
    logging.info("Starting QDA optimization (LDA optimization skipped)...")
    _, qda_summary = _optimise_classifier("qda", qda_objective, n_trials, dataset_dir, seed)
    
    # For LDA, we use fixed parameters with identity weight = 0.1
    logging.info("Using fixed LDA parameters with identity weight = 0.1")
    lda_summary = {
        "best_value": None,
        "best_params": {"reg_alpha": 0.1},
        "best_metrics": {"lda": "fixed_parameters"}
    }
    
    # Create summary
    dataset_summary = {
        "dataset": "cross_domain_concatenated",
        "datasets_used": args['cross_domain_datasets'],
        "seed": seed,
        "model": args['vit_type'],
        "iterations": args['iterations'],
        "lda": lda_summary,
        "qda": qda_summary,
    }
    
    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
    
    # Release resources
    del model
    if cache_features_on_device and eval_device.type != "cpu":
        del train_features
        del test_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logging.info(f"Optimization completed. Results saved to {dataset_dir}")
    
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