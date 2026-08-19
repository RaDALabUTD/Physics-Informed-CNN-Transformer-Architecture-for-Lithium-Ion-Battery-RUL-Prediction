import os
import urllib.request

os.makedirs("data/raw/CALCE", exist_ok=True)
calce_urls = [
    ('https://web.calce.umd.edu/batteries/data/CS2_33.zip', 'CS2_33.zip'),
    ('https://web.calce.umd.edu/batteries/data/CX2_16.zip', 'CX2_16.zip')
]

for url, fname in calce_urls:
    out = os.path.join("data/raw/CALCE", fname)
    if not os.path.exists(out):
        print(f"Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")
print("Done CALCE download.")
