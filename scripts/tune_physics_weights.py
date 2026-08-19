import argparse
import itertools

from picnnt.experiments import build_real_data, build_samsung_data, val_rmse_for_config
from picnnt.models.losses import LossWeights
from picnnt.models.pi_cnn_transformer import PICNNTransformer
from picnnt.train import TrainConfig
from picnnt.utils import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["HUST", "SAMSUNG"])
parser.add_argument("--processed-dir", default="data/processed/HUST")
parser.add_argument("--raw-dir", default="data/raw/SAMSUNG")
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = get_device()
if args.dataset == "HUST":
    data = build_real_data("HUST", args.processed_dir, stride=args.stride)
else:
    data = build_samsung_data(args.raw_dir, stride=args.stride)

mono_grid = [0.0, 0.1, 0.3]
bv_grid = [0.0, 0.05, 0.15]

best = None
for lm, lb in itertools.product(mono_grid, bv_grid):
    weights = LossWeights(lambda_mono=lm, lambda_bv=lb)
    config = TrainConfig(use_physics=(lm > 0 or lb > 0), loss_weights=weights, mono_margin=float(args.stride))
    val_rmse = val_rmse_for_config(lambda: PICNNTransformer(), data, device, config, seed=args.seed)
    print(f"lambda_mono={lm:.3f} lambda_bv={lb:.3f}  val_rmse={val_rmse:.2f}")
    if best is None or val_rmse < best[0]:
        best = (val_rmse, lm, lb)

print(f"\nbest: lambda_mono={best[1]} lambda_bv={best[2]}  val_rmse={best[0]:.2f}")
