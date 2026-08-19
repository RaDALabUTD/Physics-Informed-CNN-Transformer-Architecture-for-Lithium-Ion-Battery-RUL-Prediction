import argparse
import time

from picnnt.experiments import MODEL_REGISTRY, build_real_data, run_seeds, summarize
from picnnt.evaluate import paired_significance
from picnnt.utils import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HUST", choices=["HUST"])
parser.add_argument("--processed-dir", default="data/processed/HUST")
parser.add_argument("--n-seeds", type=int, default=5)
parser.add_argument("--seq-len", type=int, default=50)
parser.add_argument("--stride", type=int, default=20)
args = parser.parse_args()

device = get_device()
print(f"device={device}  dataset={args.dataset}  n_seeds={args.n_seeds}  stride={args.stride}")

data = build_real_data(args.dataset, args.processed_dir, seq_len=args.seq_len, stride=args.stride)
print(f"train/val/test cells: {len(data.train_ds.cell_boundaries)}/"
      f"{len(data.val_ds.cell_boundaries)}/{len(data.test_ds.cell_boundaries)}  "
      f"windows: {len(data.train_ds)}/{len(data.val_ds)}/{len(data.test_ds)}")

order = ["pi_cnn_transformer", "vanilla_transformer", "cnn", "lstm"]
all_results = {}
for name in order:
    model_fn, config = MODEL_REGISTRY[name]
    config.mono_margin = float(args.stride)
    n_params = model_fn().count_parameters()
    t0 = time.time()
    results = run_seeds(model_fn, data, device, config, n_seeds=args.n_seeds)
    summary = summarize(results)
    all_results[name] = summary
    print(f"{name:22s} RMSE={summary['rmse_mean']:.1f}±{summary['rmse_std']:.1f}  "
          f"MAE={summary['mae_mean']:.1f}±{summary['mae_std']:.1f}  "
          f"R2={summary['r2_mean']:.3f}±{summary['r2_std']:.3f}  "
          f"MAPE={summary['mape_mean']:.1f}%  params={n_params}  "
          f"({time.time()-t0:.0f}s)")

print("\nSignificance vs PI-CNN-Transformer (Welch's t-test on per-seed RMSE):")
pi_rmse = all_results["pi_cnn_transformer"]["rmse_values"]
for name in order[1:]:
    sig = paired_significance(pi_rmse, all_results[name]["rmse_values"])
    print(f"  vs {name:22s} t={sig.t_stat:.2f}  p={sig.p_value:.4f}  Cohen's d={sig.cohens_d:.2f}")
