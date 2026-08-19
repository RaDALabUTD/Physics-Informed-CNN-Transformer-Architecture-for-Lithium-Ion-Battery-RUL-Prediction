from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import torch
import torch.nn as nn

from picnnt.data.preprocessing import TemporalBatchSampler, WindowDataset, sequential_eval_batches
from picnnt.models.losses import CapacityRULCurve, LossWeights, PhysicsConstants, total_loss


@dataclasses.dataclass
class TrainConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 32
    max_epochs: int = 300
    early_stop_patience: int = 20
    lr_plateau_patience: int = 10
    lr_decay_factor: float = 0.5
    grad_clip_norm: float = 1.0
    use_physics: bool = True
    loss_weights: LossWeights = dataclasses.field(default_factory=LossWeights)
    mono_segment_len: int = 4
    mono_margin: float = 1.0


def _batched_gather(dataset: WindowDataset, indices: list[int], device: torch.device):
    xs, ys, rs, cs = zip(*[dataset[i] for i in indices])
    return (
        torch.stack(xs).to(device),
        torch.stack(ys).to(device),
        torch.stack(rs).to(device),
        torch.stack(cs).to(device),
    )


@torch.no_grad()
def _evaluate(model: nn.Module, dataset: WindowDataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    se_sum, n = 0.0, 0
    for idxs in sequential_eval_batches(dataset, batch_size):
        x, y, _, _ = _batched_gather(dataset, idxs, device)
        y_hat = model(x)
        se_sum += torch.sum((y_hat - y) ** 2).item()
        n += len(idxs)
    return math.sqrt(se_sum / max(n, 1))


def train_model(
    model: nn.Module,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    device: torch.device,
    config: TrainConfig = TrainConfig(),
    physics_constants: PhysicsConstants | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_decay_factor, patience=config.lr_plateau_patience,
    )

    batches_per_epoch = max(1, len(train_dataset) // config.batch_size)
    sampler = TemporalBatchSampler(
        train_dataset, config.batch_size, batches_per_epoch, segment_len=config.mono_segment_len,
    )
    segment_len = sampler.segment_len

    capacity_rul_curve = None
    if config.use_physics:
        if physics_constants is None:
            all_r = torch.tensor(train_dataset.resistance, dtype=torch.float32)
            physics_constants = PhysicsConstants.fit(all_r)
        capacity_rul_curve = CapacityRULCurve.fit(
            np.array(train_dataset.targets), np.array(train_dataset.capacity_fraction),
        ).to(device)

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        for idxs in sampler:
            x, y, r, c = _batched_gather(train_dataset, idxs, device)
            optimizer.zero_grad()
            y_hat = model(x)
            if config.use_physics:
                loss, _ = total_loss(
                    y_hat, y, r, physics_constants, config.loss_weights, segment_len,
                    capacity_rul_curve, config.mono_margin,
                )
            else:
                loss = torch.mean((y_hat - y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_rmse = _evaluate(model, val_dataset, device, config.batch_size)
        scheduler.step(val_rmse)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train_loss={history['train_loss'][-1]:.4f}  val_rmse={val_rmse:.3f}")

        if epochs_without_improvement >= config.early_stop_patience:
            if verbose:
                print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    history["best_val_rmse"] = best_val_rmse
    history["epochs_trained"] = epoch + 1
    return model, history
