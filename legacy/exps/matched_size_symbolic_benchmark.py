"""Matched-size symbolic benchmark.

This robustness check keeps the main benchmark protocol unchanged but selects,
for each model family, the candidate width whose trainable parameter count is
closest to a target size.  It is intended for small high-complexity checks such
as c=90 rather than a full scaling-law sweep.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import MODEL_REGISTRY, TRANSFORMER_STYLE_MODELS, get_model
from symbolic_sequence_benchmark import generate_seed_strings, prepare_data, train_and_evaluate


LOGGER = logging.getLogger("matched-size-symbolic")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LSTM", "GRU", "Transformer", "LinearAttention", "Performer", "RWKV"],
    )
    parser.add_argument("--target-sizes-m", nargs="+", type=float, default=[0.2])
    parser.add_argument("--symbols", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--complexities", nargs="+", type=int, default=[90])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[3500])
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--forecast-horizon", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--layers", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--recurrent-units",
        nargs="+",
        type=int,
        default=[32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768],
        help="Candidate hidden widths for recurrent-style models.",
    )
    parser.add_argument(
        "--d-models",
        nargs="+",
        type=int,
        default=[32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256],
        help="Candidate d_model values for Transformer-style models.",
    )
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--optimizers", nargs="+", choices=["Adam", "AdamW"], default=["AdamW"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.01])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stopping-loss", type=float, default=0.05)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--output-dir", default="exps/results_symbolic_matched_size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--task-count", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--dry-run", action="store_true", help="Only write matched_configs.csv and exit.")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny configuration for installation checks.")
    return parser.parse_args()


def configure(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    if args.smoke:
        args.models = ["LSTM", "GRU", "Transformer", "RWKV"]
        args.target_sizes_m = [0.05]
        args.symbols = [2]
        args.complexities = [10]
        args.sequence_lengths = [320]
        args.seed_count = 1
        args.runs = 1
        args.layers = [1]
        args.recurrent_units = [16, 32]
        args.d_models = [16, 32]
        args.batch_sizes = [32]
        args.max_epochs = 2
        args.patience = 1
        args.learning_rates = [1e-3]
        args.weight_decays = [0.0]
        args.task_index = 0
        args.task_count = 1
    if args.task_count < 1:
        raise ValueError("--task-count must be at least 1")
    if args.task_index < 0 or args.task_index >= args.task_count:
        raise ValueError("--task-index must satisfy 0 <= task_index < task_count")
    for unknown in set(args.models) - set(MODEL_REGISTRY):
        raise ValueError(f"Unknown model '{unknown}'. Available: {sorted(MODEL_REGISTRY)}")


def iter_candidate_shapes(model_name, args):
    if model_name in TRANSFORMER_STYLE_MODELS:
        for d_model in args.d_models:
            yield {"units": args.ff_mult * d_model, "d_model": d_model}
    else:
        for units in args.recurrent_units:
            yield {"units": units, "d_model": 0}


def count_model_params(model_name, alphabet_size, num_layers, units, d_model, window_size):
    model = get_model(
        model_name,
        alphabet_size,
        units,
        alphabet_size,
        num_layers=num_layers,
        d_model=max(1, d_model),
        window_size=window_size,
    )
    n_params = sum(param.numel() for param in model.parameters())
    del model
    return n_params


def select_matched_config(model_name, alphabet_size, num_layers, target_size_m, args):
    target_params = target_size_m * 1e6
    candidates = []
    for shape in iter_candidate_shapes(model_name, args):
        try:
            n_params = count_model_params(
                model_name,
                alphabet_size,
                num_layers,
                shape["units"],
                shape["d_model"],
                args.window_size,
            )
        except ImportError as exc:
            raise ImportError(f"Cannot instantiate {model_name}; missing dependency.") from exc
        candidates.append({**shape, "model_params": n_params, "model_size_m": n_params / 1e6})
    if not candidates:
        raise ValueError(f"No candidate shapes for {model_name}")
    return min(candidates, key=lambda item: abs(item["model_params"] - target_params))


def build_matched_config_table(args):
    rows = []
    for symbols in args.symbols:
        for target_size_m in args.target_sizes_m:
            for layer in args.layers:
                for model_name in args.models:
                    config = select_matched_config(model_name, symbols, layer, target_size_m, args)
                    error_m = abs(config["model_size_m"] - target_size_m)
                    rows.append(
                        {
                            "model": model_name,
                            "symbols": symbols,
                            "layers": layer,
                            "matched_target_size_m": target_size_m,
                            "units": config["units"],
                            "d_model": config["d_model"],
                            "model_params": config["model_params"],
                            "model_size_m": config["model_size_m"],
                            "matched_param_abs_error_m": error_m,
                            "matched_param_rel_error": error_m / max(target_size_m, 1e-12),
                        }
                    )
    return pd.DataFrame(rows)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    matched_configs = build_matched_config_table(args)
    matched_configs.to_csv(output_dir / "matched_configs.csv", index=False)
    LOGGER.info("Matched configs:\n%s", matched_configs.to_string(index=False))
    if args.dry_run:
        return matched_configs

    rows = []
    results_name = "results.csv" if args.task_count == 1 else f"results_task_{args.task_index:03d}.csv"
    results_path = output_dir / results_name
    total_configs = 0
    selected_configs = 0
    device = torch.device(args.device)

    config_lookup = {
        (row.model, int(row.symbols), int(row.layers), float(row.matched_target_size_m)): row
        for row in matched_configs.itertuples(index=False)
    }

    for symbols in args.symbols:
        LOGGER.info("Generating seeds for alphabet size %s", symbols)
        seed_df = generate_seed_strings(symbols, args.complexities, args.seed_count, args.seed)
        for complexity in args.complexities:
            target_complexity = int(complexity)
            measured = seed_df["LZW_complexity"].astype(int)
            subset = seed_df[measured == target_complexity]
            if subset.empty:
                LOGGER.warning("No exact seed for symbols=%s target_complexity=%s", symbols, target_complexity)
                continue
            for seed_idx, row in subset.head(args.seed_count).reset_index(drop=True).iterrows():
                seed_string = str(row["string"])
                measured_complexity = int(row["LZW_complexity"])
                for target_size_m in args.target_sizes_m:
                    for model_name in args.models:
                        for layer in args.layers:
                            matched = config_lookup[(model_name, symbols, layer, float(target_size_m))]
                            hidden_size = int(matched.units)
                            d_model = int(matched.d_model)
                            n_params = int(matched.model_params)
                            for sequence_length in args.sequence_lengths:
                                for optimizer_name in args.optimizers:
                                    for learning_rate in args.learning_rates:
                                        for weight_decay in args.weight_decays:
                                            for batch_size in args.batch_sizes:
                                                for run_idx in range(args.runs):
                                                    config_index = total_configs
                                                    total_configs += 1
                                                    if config_index % args.task_count != args.task_index:
                                                        continue
                                                    selected_configs += 1
                                                    data = prepare_data(
                                                        seed_string,
                                                        sequence_length,
                                                        args.window_size,
                                                        args.forecast_horizon,
                                                        device,
                                                    )
                                                    torch.manual_seed(args.seed + run_idx)
                                                    model = get_model(
                                                        model_name,
                                                        len(data["symbols"]),
                                                        hidden_size,
                                                        len(data["symbols"]),
                                                        num_layers=layer,
                                                        d_model=max(1, d_model),
                                                        window_size=args.window_size,
                                                    ).to(device)
                                                    if args.device == "cuda":
                                                        torch.cuda.reset_peak_memory_stats(device)
                                                    LOGGER.info(
                                                        "Training %s target=%.3fM params=%.3fM C=%s A=%s N=%s L=%s U=%s D=%s run=%s",
                                                        model_name,
                                                        target_size_m,
                                                        n_params / 1e6,
                                                        measured_complexity,
                                                        symbols,
                                                        sequence_length,
                                                        layer,
                                                        hidden_size,
                                                        d_model,
                                                        run_idx,
                                                    )
                                                    try:
                                                        metrics = train_and_evaluate(
                                                            model,
                                                            data,
                                                            args,
                                                            optimizer_name,
                                                            learning_rate,
                                                            weight_decay,
                                                            batch_size,
                                                        )
                                                    except Exception as exc:
                                                        LOGGER.exception("Run failed: %s", exc)
                                                        continue
                                                    result = {
                                                        "config_index": config_index,
                                                        "experiment": "matched_size_high_complexity",
                                                        "model": model_name,
                                                        "symbols": symbols,
                                                        "target_complexity": target_complexity,
                                                        "complexity": measured_complexity,
                                                        "measured_lzw_complexity": measured_complexity,
                                                        "seed_index": seed_idx,
                                                        "seed_string": seed_string,
                                                        "target_string": data["target"],
                                                        "sequence_length": data["sequence_length"],
                                                        "window_size": args.window_size,
                                                        "forecast_horizon": args.forecast_horizon,
                                                        "layers": layer,
                                                        "units": hidden_size,
                                                        "d_model": d_model,
                                                        "matched_target_size_m": target_size_m,
                                                        "matched_param_abs_error_m": float(matched.matched_param_abs_error_m),
                                                        "matched_param_rel_error": float(matched.matched_param_rel_error),
                                                        "optimizer": optimizer_name,
                                                        "learning_rate": learning_rate,
                                                        "weight_decay": weight_decay,
                                                        "batch_size": batch_size,
                                                        "run": run_idx,
                                                        "model_params": n_params,
                                                        "model_size_m": n_params / 1e6,
                                                        **metrics,
                                                    }
                                                    rows.append(result)
                                                    pd.DataFrame(rows).to_csv(results_path, index=False)
                                                    LOGGER.info(
                                                        "%s loss=%.4f acc=%.3f DL=%.3f JW=%.3f",
                                                        model_name,
                                                        result["test_loss"],
                                                        result["test_accuracy"],
                                                        result["DL"],
                                                        result["JW"],
                                                    )
    LOGGER.info("Selected %s/%s configurations for this task", selected_configs, total_configs)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parsed = parse_args()
    configure(parsed)
    run(parsed)
