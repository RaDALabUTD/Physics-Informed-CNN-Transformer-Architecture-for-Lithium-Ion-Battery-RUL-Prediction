import os
import urllib.request
import zipfile

os.makedirs("data/raw/NASA", exist_ok=True)
os.makedirs("data/raw/RWTH", exist_ok=True)

# NASA PCoE
nasa_url = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
nasa_zip = "data/raw/NASA/Battery_Data_Set.zip"
if not os.path.exists(nasa_zip):
    print("Downloading NASA...")
    urllib.request.urlretrieve(nasa_url, nasa_zip)
    print("Unzipping NASA...")
    with zipfile.ZipFile(nasa_zip, 'r') as zip_ref:
        zip_ref.extractall("data/raw/NASA")
else:
    print("NASA already downloaded.")

# RWTH
rwth_url = "https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip"
rwth_zip = "data/raw/RWTH/RWTH.zip"
if not os.path.exists(rwth_zip):
    print("Downloading RWTH...")
    urllib.request.urlretrieve(rwth_url, rwth_zip)
else:
    print("RWTH already downloaded.")

print("Done.")
