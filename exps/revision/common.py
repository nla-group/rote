"""Protocol-compatible training and durable stage outputs for the revision."""

from dataclasses import asdict, dataclass
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from rote_benchmark.benchmark import (evaluate_loader, make_optimizer, prepare_data,
                                      rollout, train_and_evaluate)
from rote_benchmark.models import TRANSFORMER_STYLE_MODELS, get_model

from .seeds import phase_statistics, predictive_order


@dataclass(frozen=True)
class Protocol:
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 10
    stopping_loss: float = 0.05
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    weight_decay: float = 0.01
    learning_rate: float = 0.0003
    optimizer: str = "AdamW"
    seed: int = 3407
    device: str = "cuda"
    layers: int = 2
    units: int = 128
    d_model: int = 256
    horizon: int = 100


def setup(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a smoke test")
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def effective_hidden_size(model_name, protocol):
    """Match the main benchmark's 4*d_model feed-forward width convention."""
    return (4 * protocol.d_model if model_name in TRANSFORMER_STYLE_MODELS
            else protocol.units)


def train_with_warmup(model, data, args, optimizer_name, learning_rate,
                      weight_decay, batch_size, warmup_epochs):
    """Original trainer with only a linear epoch-level LR warmup added."""
    train_loader = DataLoader(TensorDataset(data["X_train"], data["y_train"]),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(data["X_valid"], data["y_valid"]),
                              batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(data["X_test"], data["y_test"]),
                             batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    best_valid, best_state, bad_epochs, epochs_used = float("inf"), None, 0, 0
    start = time.time()
    for epoch in range(args.max_epochs):
        if epoch < warmup_epochs:
            rate = learning_rate * (epoch + 1) / warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = rate
        elif epoch == warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Invalid training loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        if epoch >= warmup_epochs:
            scheduler.step()
        valid_loss, _ = evaluate_loader(model, valid_loader, criterion)
        epochs_used = epoch + 1
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if valid_loss <= args.stopping_loss or bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate_loader(model, test_loader, criterion)
    roll = rollout(model, data["initial_context"], data["target"], data["symbols"],
                   args.window_size, torch.device(args.device))
    elapsed = time.time() - start
    device = torch.device(args.device)
    return {
        "train_time": elapsed, "epochs": epochs_used,
        "time_per_epoch": elapsed / max(1, epochs_used),
        "memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2
                      if device.type == "cuda" else 0.0),
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "DL": roll["dl"], "JW": roll["jw"], "forecast": roll["forecast"],
    }


def fit_one(model_name, row, *, window=100, sequence_length=3500, run=0,
            protocol=Protocol(), learning_rate=None, warmup_epochs=0,
            periodic_diagnostics=True, track=""):
    """Reuse the manuscript's model, split, optimizer, stopping, and rollout.

    train/validation metrics are evaluated after the original trainer restores
    its best-validation checkpoint. The training routine is not modified.
    """
    seed_string = str(row["seed_string"])
    random_seed = protocol.seed + run
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    data = prepare_data(seed_string, sequence_length, window, protocol.horizon,
                        protocol.device)
    n = len(data["symbols"])
    hidden_size = effective_hidden_size(model_name, protocol)
    model = get_model(model_name, n, hidden_size, n, num_layers=protocol.layers,
                      d_model=protocol.d_model, window_size=window).to(protocol.device)
    if protocol.device == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(protocol.device))
    args = SimpleNamespace(
        max_epochs=protocol.max_epochs, patience=protocol.patience,
        stopping_loss=protocol.stopping_loss, lr_step_size=protocol.lr_step_size,
        lr_gamma=protocol.lr_gamma, window_size=window, device=protocol.device,
    )
    rate = protocol.learning_rate if learning_rate is None else learning_rate
    trainer = train_with_warmup if warmup_epochs else train_and_evaluate
    trainer_args = (model, data, args, protocol.optimizer, rate,
                    protocol.weight_decay, protocol.batch_size)
    result = (trainer(*trainer_args, warmup_epochs)
              if warmup_epochs else trainer(*trainer_args))
    criterion = nn.CrossEntropyLoss()
    diagnostics = {}
    for split in ("train", "valid"):
        loader = DataLoader(TensorDataset(data[f"X_{split}"], data[f"y_{split}"]),
                            batch_size=protocol.batch_size)
        loss, accuracy = evaluate_loader(model, loader, criterion)
        diagnostics[f"{split}_loss"] = loss
        diagnostics[f"{split}_accuracy"] = accuracy
    diagnostics["reached_stopping_loss"] = int(
        diagnostics["valid_loss"] <= protocol.stopping_loss
    )
    order = predictive_order(seed_string) if periodic_diagnostics else None
    phase = phase_statistics(seed_string, window) if periodic_diagnostics else None
    result.update(diagnostics)
    result.update({
        "track": track, "model": model_name, "symbols": n,
        "complexity": (int(row["complexity"]) if row["complexity"] is not None else None),
        "seed_index": int(row["seed_index"]),
        "seed_string": seed_string, "seed_length": len(seed_string),
        "predictive_order": order, "alpha_w": phase["alpha"] if phase else None,
        "bayes_error_w": phase["bayes_error"] if phase else None,
        "conditional_entropy_w": phase["conditional_entropy"] if phase else None,
        "sequence_length": data["sequence_length"], "window_size": window,
        "forecast_horizon": protocol.horizon, "run": run,
        "optimizer": protocol.optimizer, "learning_rate": rate,
        "warmup_epochs": warmup_epochs,
        "weight_decay": protocol.weight_decay, "batch_size": protocol.batch_size,
        "layers": protocol.layers, "units": hidden_size,
        "d_model": protocol.d_model, "random_seed": random_seed,
        "model_params": sum(p.numel() for p in model.parameters()),
        "train_windows": len(data["X_train"]),
        "valid_windows": len(data["X_valid"]),
        "test_windows": len(data["X_test"]),
        "phase_exposure": (len(data["X_train"]) / len(seed_string)
                           if periodic_diagnostics else None),
        "target_string": data["target"],
        "exact": int(data["target"] == result["forecast"]),
    })
    return result


def stage_config(stage, protocol, **extra):
    return {"stage": stage, "protocol": asdict(protocol), **extra}
