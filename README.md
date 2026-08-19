# Physics-Informed CNN-Transformer for Battery RUL Prediction

> **Note:** This repository is under active development toward a more extended, rigorously validated version of this work. Code, datasets, and results are preliminary and subject to change. The core PI-CNN-Transformer architecture and training pipeline are reproducible and usable as-is, although it may have had additions to it in terms of attributes.

This repository contains an implementation of a Physics-Informed CNN-Transformer (PI-CNN-T) architecture for remaining-useful-life (RUL) prediction of lithium-ion batteries, combining a convolutional front-end, a Transformer encoder, and physics-informed regularization terms derived from Butler-Volmer electrochemical kinetics.

Conventional data-driven RUL models learn purely from historical capacity-fade trajectories and have no mechanism to respect the underlying electrochemistry of degradation. This implementation instead regularizes the model's predictions against a physically-derived bound relating internal cell resistance to remaining capacity, so that predictions are constrained by known electrochemical behavior rather than fit to data alone.

## Method Overview

The architecture integrates three components:

**CNN front-end.** A 1D convolutional stack (32→64 channels) extracts local temporal features from raw per-cycle sequences (voltage, current, temperature, capacity, internal resistance) before they reach the encoder.

**Transformer encoder.** A multi-head self-attention encoder with fixed sinusoidal positional encoding models long-range dependencies across the cycle history, producing a pooled representation used for RUL regression.

**Physics-informed loss.** Two auxiliary loss terms are added to the standard regression loss:
- *Monotonicity loss* penalizes locally non-monotonic RUL predictions across nearby cycles, reflecting the fact that RUL should decrease as cycling progresses.
- *Butler-Volmer loss* penalizes predicted capacity that exceeds a physically-derived ceiling, computed from measured internal resistance via Butler-Volmer kinetics and an empirical (isotonic-regression-fit) capacity-RUL relationship.

## Core Novelty

The main contribution is coupling a data-driven capacity-RUL relationship, fit directly from training data via isotonic regression, with a resistance-derived physical ceiling from Butler-Volmer kinetics, and using that combination as a soft constraint during training rather than a hard rule.


## Implementation Notes

**Architecture.** The CNN front-end and Transformer encoder follow the same overall structure described in Method Overview. Internal resistance is included directly as a fifth model input feature alongside voltage, current, temperature, and capacity, rather than being used only inside the physics loss. Reduced-capacity variants of both the physics-informed and vanilla models (smaller model dimension, fewer encoder layers) are included for comparing performance against parameter count on smaller datasets.

**Loss formulation.** Two variants of the Butler-Volmer term are implemented: the original one-sided hinge, which only penalizes predictions that imply capacity exceeding the physics-derived ceiling, and a symmetric variant that pulls predictions toward the physics-implied capacity in both directions. The one-sided version can go structurally inactive on datasets where predictions never exceed the ceiling; the symmetric version guarantees a live gradient signal regardless.

**Training infrastructure.** Deterministic, reproducible training (fixed seeding across cuDNN and PyTorch), temporal batch sampling that draws multi-cell batches while preserving within-cell cycle ordering (needed for BatchNorm to behave correctly and for the monotonicity loss to have meaningful within-batch segments), per-dataset handling of internal resistance (real measured values where available, an estimated proxy otherwise), and ablation tooling to isolate the contribution of individual architectural and physics-informed loss components.

## Further Plans

A few directions currently being worked through:

- **Resistance-quality-aware weighting.** Rather than a fixed Butler-Volmer loss weight applied uniformly, weighting by an estimate of resistance-signal quality (real measurement vs. proxy, and how clean the proxy is) would turn the current empirical finding into an explicit mechanism.
- **Calibrated uncertainty quantification.** The current setup does not yet include a properly calibrated uncertainty estimate; replacing naive dropout-based sampling with a calibrated (e.g. split-conformal) approach is planned.
- **Broader dataset coverage.** Additional real cycling datasets with measured or estimated resistance are being integrated to test how far the current findings generalize, including datasets still pending data access.
- **Cross-chemistry evaluation.** With LFP, NMC, and LCO cells now represented across the supported datasets, a systematic cross-chemistry generalization study is planned.

## Ongoing Work

This codebase is under active development — it is being continually extended with new datasets, refined loss formulations, and additional ablation studies as the work progresses.

Current development is centered on characterizing the specific conditions under which Butler-Volmer regularization provides measurable benefit. The evidence so far, from ablation on real data:

- **MATR** (147 cells, real measured resistance): removing Butler-Volmer costs a reproducible ~7% increase in RMSE.
- **RWTH** (48 cells, IR-drop proxy resistance): removing Butler-Volmer costs ~5% RMSE (55.7 vs. 53.0); removing monotonicity costs ~4%; removing the CNN front-end costs ~8% — every component contributes on this dataset.
- Resistance signal *quality*, not simply real-vs-proxy, looks like the more precise variable: RWTH's proxy is computed under a simple constant-current protocol and shows a real benefit, while a similarly-computed proxy under HUST's more complex fast-charging protocol shows less benefit.

## Applications

- Battery-pack health monitoring and RUL estimation for EVs and grid storage
- Predictive maintenance scheduling for battery-backed systems
- Research into physics-informed regularization for degradation modeling more broadly

## Citation

If you use this work, please cite:

Ganji, A., Zhou, Y., and Xu, Y. Physics-Informed CNN-Transformer Architecture for Lithium-Ion Battery Remaining Useful Life Prediction. Proceedings of the ASME 2026 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference (IDETC-CIE2026), IDETC2026-193995, Houston, TX.
