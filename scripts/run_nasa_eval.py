import torch
import sys
from pathlib import Path
sys.path.append(str(Path("src").absolute()))

from picnnt.experiments import build_nasa_data, MODEL_REGISTRY, run_seeds, summarize
from picnnt.train import TrainConfig
from picnnt.models.losses import LossWeights

print("Loading NASA data...")
data = build_nasa_data("data/raw/NASA")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fn, config = MODEL_REGISTRY["pi_cnn_transformer"]

print("Running PI-CNN-Transformer on NASA...")
results = run_seeds(model_fn, data, device, config, n_seeds=3)
summary = summarize(results)
print("NASA PI-CNN-Transformer Metrics:", summary)
