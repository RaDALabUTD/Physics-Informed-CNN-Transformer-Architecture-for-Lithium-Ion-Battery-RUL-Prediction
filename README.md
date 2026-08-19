# Physics-Informed CNN-Transformer for Battery RUL Prediction

This repository contains an implementation of a Physics-Informed CNN-Transformer (PI-CNN-T) architecture for remaining-useful-life (RUL) prediction of lithium-ion batteries, combining a convolutional front-end, a Transformer encoder, and physics-informed regularization terms derived from Butler-Volmer electrochemical kinetics.

Conventional data-driven RUL models learn purely from historical capacity-fade trajectories and have no mechanism to respect the underlying electrochemistry of degradation. This implementation instead regularizes the model's predictions against a physically-derived bound relating internal cell resistance to remaining capacity, so that predictions are constrained by known electrochemical behavior rather than fit to data alone.

## Method Overview

The architecture integrates three components:

**CNN front-end.** A 1D convolutional stack (32→64 channels) extracts local temporal features from raw per-cycle sequences (voltage, current, temperature, capacity, internal resistance) before they reach the encoder.

**Transformer encoder.** A multi-head self-attention encoder with learned positional encoding models long-range dependencies across the cycle history, producing a pooled representation used for RUL regression.

**Physics-informed loss.** Two auxiliary loss terms are added to the standard regression loss:
- *Monotonicity loss* penalizes locally non-monotonic RUL predictions across nearby cycles, reflecting the fact that RUL should decrease as cycling progresses.
- *Butler-Volmer loss* penalizes predicted capacity that exceeds a physically-derived ceiling, computed from measured internal resistance via Butler-Volmer kinetics and an empirical (isotonic-regression-fit) capacity-RUL relationship.

## Core Novelty

The main contribution is coupling a data-driven capacity-RUL relationship, fit directly from training data via isotonic regression, with a resistance-derived physical ceiling from Butler-Volmer kinetics, and using that combination as a soft constraint during training rather than a hard rule.

## Implementation Notes

This implementation includes deterministic, reproducible training (fixed seeding across cuDNN and PyTorch), per-dataset handling of internal resistance (real measured values where available, an estimated proxy otherwise), temporal batch sampling to preserve within-cell ordering, and ablation tooling to isolate the contribution of individual architectural and physics-informed loss components.


## Ongoing Work

This codebase is under active development — it is being continually extended with new datasets, refined loss formulations, and additional ablation studies as the work progresses.

Current development is centered on characterizing the specific conditions under which Butler-Volmer regularization provides measurable benefit. The finding so far, verified via ablation on real data: the benefit is conditional on real measured resistance data being available. On MATR (147 cells, real measured internal resistance), removing the Butler-Volmer term costs a reproducible ~7% increase in RMSE. On datasets where internal resistance is only an estimated proxy rather than a direct measurement, the term shows less reliable benefit. This is being further tested on NASA PCoE (real EIS-derived charge-transfer resistance) and additional datasets as they're integrated, to determine how general this pattern is.

## Applications

- Battery-pack health monitoring and RUL estimation for EVs and grid storage
- Predictive maintenance scheduling for battery-backed systems
- Research into physics-informed regularization for degradation modeling more broadly

## Citation

If you use this work, please cite:

Ganji, A., Zhou, Y., and Xu, Y. Physics-Informed CNN-Transformer Architecture for Lithium-Ion Battery Remaining Useful Life Prediction. Proceedings of the ASME 2026 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference (IDETC-CIE2026), IDETC2026-193995, Houston, TX.
