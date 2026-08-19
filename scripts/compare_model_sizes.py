import argparse
import time

from picnnt.experiments import (
    MODEL_REGISTRY, build_nasa_data, build_real_data, build_samsung_data, build_zhang_data, run_seeds, summarize,
)
from picnnt.utils import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["HUST", "MATR", "SAMSUNG", "NASA", "ZHANG", "RWTH"])
parser.add_argument("--processed-dir", default=None)
parser.add_argument("--raw-dir", default=None)
parser.add_argument("--n-seeds", type=int, default=2)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--seq-len", type=int, default=50)
args = parser.parse_args()

device = get_device()
if args.dataset == "SAMSUNG":
    data = build_samsung_data(args.raw_dir or "data/raw/SAMSUNG", stride=args.stride, seq_len=args.seq_len)
elif args.dataset == "NASA":
    data = build_nasa_data(
        args.raw_dir or "data/raw/NASA/5. Battery Data Set", stride=args.stride, seq_len=args.seq_len,
    )
elif args.dataset == "ZHANG":
    data = build_zhang_data(args.raw_dir or "data/raw/ZHANG_EIS/extracted", stride=args.stride, seq_len=args.seq_len)
else:
    processed_dir = args.processed_dir or f"data/processed/{args.dataset}"
    data = build_real_data(args.dataset, processed_dir, stride=args.stride, seq_len=args.seq_len)

print(f"train/val/test cells: {len(data.train_ds.cell_boundaries)}/"
      f"{len(data.val_ds.cell_boundaries)}/{len(data.test_ds.cell_boundaries)}  "
      f"windows: {len(data.train_ds)}/{len(data.val_ds)}/{len(data.test_ds)}")

order = ["pi_cnn_transformer", "pi_cnn_transformer_small", "vanilla_transformer", "vanilla_transformer_small"]
for name in order:
    model_fn, config = MODEL_REGISTRY[name]
    config.mono_margin = float(args.stride)
    n_params = model_fn().count_parameters()
    t0 = time.time()
    results = run_seeds(model_fn, data, device, config, n_seeds=args.n_seeds)
    s = summarize(results)
    print(f"{name:28s} RMSE={s['rmse_mean']:.1f}±{s['rmse_std']:.1f}  R2={s['r2_mean']:.3f}±{s['r2_std']:.3f}  "
          f"params={n_params}  ({time.time()-t0:.0f}s)")
