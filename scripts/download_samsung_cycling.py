import time
from pathlib import Path

import requests

OUT_DIR = Path("data/raw/SAMSUNG")
OUT_DIR.mkdir(parents=True, exist_ok=True)

urls_file = Path(__file__).parent / "samsung_urls.txt"
lines = urls_file.read_text().strip().splitlines()
print(f"{len(lines)} files to download")

for i, line in enumerate(lines):
    name, url = line.split("\t")
    out_path = OUT_DIR / f"{name}.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[{i+1}/{len(lines)}] {name} already exists, skipping")
        continue
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            size_mb = out_path.stat().st_size / 1e6
            print(f"[{i+1}/{len(lines)}] {name}: {size_mb:.1f}MB")
            break
        except Exception as e:
            print(f"  attempt {attempt+1} failed for {name}: {e}")
            time.sleep(3)
    else:
        print(f"  GAVE UP on {name}")

print("done")
