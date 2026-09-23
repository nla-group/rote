"""Controlled symbolic next-token benchmark.

This script implements the experiment protocol used by the transformer
benchmark manuscript: generate LZW-controlled symbolic seeds, repeat each seed
to form a periodic sequence, train finite-context next-symbol predictors, and
evaluate both held-out next-token accuracy and recursive rollout quality.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slearn import lzw_string_seeds
from slearn.dmetric import (
    normalized_damerau_levenshtein_distance,
    normalized_jaro_winkler_distance,
)
from models import MODEL_REGISTRY, get_model


LOGGER = logging.getLogger("symbolic-benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LSTM", "GRU", "minGRU", "minLSTM", "Transformer", "LinearAttention", "Performer", "RWKV"],
    )
    parser.add_argument("--symbols", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--complexities", nargs="+", type=int, default=[10, 30, 50, 70, 90])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[3500])
    parser.add_argument("--token-ratios", nargs="*", type=float, default=[])
    parser.add_argument("--max-sequence-length", type=int, default=20000)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--forecast-horizon", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--layers", nargs="+", type=int, default=[2])
    parser.add_argument("--units", nargs="+", type=int, default=[128])
    parser.add_argument("--d-models", nargs="+", type=int, default=[256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--optimizers", nargs="+", choices=["Adam", "AdamW"], default=["AdamW"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.01])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stopping-loss", type=float, default=0.05)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--output-dir", default="exps/results_symbolic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--task-count", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
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
        args.models = ["LSTM", "GRU", "minGRU", "minLSTM", "LinearAttention", "Performer", "RWKV"]
        args.symbols = [2]
        args.complexities = [10]
        args.sequence_lengths = [320]
        args.seed_count = 1
        args.runs = 1
        args.layers = [1]
        args.units = [32]
        args.d_models = [64]
        args.batch_sizes = [32]
        args.learning_rates = [1e-3]
        args.weight_decays = [0.0]
        args.max_epochs = 2
        args.patience = 1
        args.task_index = 0
        args.task_count = 1
    if args.task_count < 1:
        raise ValueError("--task-count must be at least 1")
    if args.task_index < 0 or args.task_index >= args.task_count:
        raise ValueError("--task-index must satisfy 0 <= task_index < task_count")


def one_hot(indices, alphabet_size):
    return F.one_hot(indices, num_classes=alphabet_size).float()


def prepare_data(seed_string, sequence_length, window_size, forecast_horizon, device):
    min_length = window_size + forecast_horizon + 2
    if sequence_length < min_length:
        sequence_length = min_length
    repeats = int(np.ceil(sequence_length / len(seed_string)))
    sequence = (seed_string * repeats)[:sequence_length]
    symbols = sorted(set(seed_string))
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    encoded = torch.tensor([symbol_to_idx[s] for s in sequence], dtype=torch.long)

    target = sequence[-forecast_horizon:]
    initial_context = sequence[-forecast_horizon - window_size : -forecast_horizon]
    train_prefix = encoded[:-forecast_horizon]

    X, y = [], []
    for i in range(len(train_prefix) - window_size):
        X.append(train_prefix[i : i + window_size])
        y.append(train_prefix[i + window_size])
    if not X:
        raise ValueError("No training windows generated; increase sequence_length.")

    X = torch.stack(X)
    y = torch.stack(y)
    n = len(X)
    train_end = max(1, int(0.8 * n))
    valid_end = max(train_end + 1, int(0.9 * n))
    valid_end = min(valid_end, n - 1)

    X_train = one_hot(X[:train_end], len(symbols)).to(device)
    y_train = y[:train_end].to(device)
    X_valid = one_hot(X[train_end:valid_end], len(symbols)).to(device)
    y_valid = y[train_end:valid_end].to(device)
    X_test = one_hot(X[valid_end:], len(symbols)).to(device)
    y_test = y[valid_end:].to(device)
    if len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError("Validation or test split is empty; increase sequence_length.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "symbols": symbols,
        "target": target,
        "initial_context": initial_context,
        "sequence_length": sequence_length,
    }


def make_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.numel()
    return total_loss / max(1, len(loader)), correct / max(1, total)


def rollout(model, initial_context, target, symbols, window_size, device):
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    context = list(initial_context)
    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(len(target)):
            indices = torch.tensor([[symbol_to_idx[s] for s in context[-window_size:]]], device=device)
            X = one_hot(indices, len(symbols))
            logits = model(X)
            next_symbol = symbols[logits.argmax(dim=-1).item()]
            generated.append(next_symbol)
            context.append(next_symbol)
    forecast = "".join(generated)
    return {
        "forecast": forecast,
        "dl": normalized_damerau_levenshtein_distance(target, forecast),
        "jw": normalized_jaro_winkler_distance(target, forecast),
    }


def train_and_evaluate(model, data, args, optimizer_name, learning_rate, weight_decay, batch_size):
    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(data["X_valid"], data["y_valid"]), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(data["X_test"], data["y_test"]), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    start = time.time()
    best_valid = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_used = 0

    for epoch in range(args.max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Invalid training loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        scheduler.step()

        valid_loss, _ = evaluate_loader(model, valid_loader, criterion)
        epochs_used = epoch + 1
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if valid_loss <= args.stopping_loss or bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate_loader(model, test_loader, criterion)
    roll = rollout(
        model,
        data["initial_context"],
        data["target"],
        data["symbols"],
        args.window_size,
        torch.device(args.device),
    )
    elapsed = time.time() - start
    device = torch.device(args.device)
    memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return {
        "train_time": elapsed,
        "epochs": epochs_used,
        "time_per_epoch": elapsed / max(1, epochs_used),
        "memory_mb": memory,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "DL": roll["dl"],
        "JW": roll["jw"],
        "forecast": roll["forecast"],
    }


def generate_seed_strings(symbols, complexities, seed_count, seed):
    df = lzw_string_seeds(
        symbols=symbols,
        complexity=complexities,
        iterations=seed_count,
        priorise_complexity=True,
        random_state=seed,
    )
    df = df.dropna(subset=["string"])
    return df[df["nr_symbols"] == symbols]


def model_sequence_lengths(base_lengths, token_ratios, n_params, max_sequence_length):
    lengths = set(base_lengths)
    for ratio in token_ratios:
        lengths.add(min(max_sequence_length, max(1, int(np.ceil(ratio * n_params)))))
    return sorted(lengths)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    results_name = "results.csv" if args.task_count == 1 else f"results_task_{args.task_index:03d}.csv"
    results_path = output_dir / results_name
    total_configs = 0
    selected_configs = 0

    device = torch.device(args.device)
    for unknown in set(args.models) - set(MODEL_REGISTRY):
        raise ValueError(f"Unknown model '{unknown}'. Available: {sorted(MODEL_REGISTRY)}")

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
                for model_name in args.models:
                    for layer in args.layers:
                        for unit in args.units:
                            for d_model in args.d_models:
                                hidden_size = 4 * d_model if model_name in {"Transformer", "BERT", "GPT", "LinearAttention", "Performer", "RWKV"} else unit
                                model = get_model(
                                    model_name,
                                    symbols,
                                    hidden_size,
                                    symbols,
                                    num_layers=layer,
                                    d_model=d_model,
                                    window_size=args.window_size,
                                )
                                n_params = sum(p.numel() for p in model.parameters())
                                del model
                                lengths = model_sequence_lengths(
                                    args.sequence_lengths,
                                    args.token_ratios,
                                    n_params,
                                    args.max_sequence_length,
                                )
                                for sequence_length in lengths:
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
                                                            d_model=d_model,
                                                            window_size=args.window_size,
                                                        ).to(device)
                                                        if args.device == "cuda":
                                                            torch.cuda.reset_peak_memory_stats(device)
                                                        LOGGER.info(
                                                            "Training %s C=%s A=%s N=%s L=%s U=%s D=%s run=%s",
                                                            model_name,
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
