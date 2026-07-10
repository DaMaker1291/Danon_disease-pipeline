import urllib.request, os, json, zipfile
base = "C:/Users/supro/Downloads/life/data"

# AlphaSeq - fix path
print("Downloading AlphaSeq Antibody Dataset...")
url = "https://zenodo.org/api/records/7783546"
try:
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    for f in data.get("files", []):
        if f["key"].endswith(".zip"):
            fname = f"{base}/AlphaSeq.zip"
            print(f"  Downloading {f['key']} ({f['size']/1024/1024:.1f} MB)...")
            urllib.request.urlretrieve(f["links"]["self"], fname)
            print(f"  Extracting...")
            with zipfile.ZipFile(fname, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.csv'):
                        z.extract(name, base)
                        print(f"  Extracted: {name}")
            os.remove(fname)
except Exception as e:
    print(f"  AlphaSeq failed: {e}")

# AAV2 Capsid - try HuggingFace API
print("\nSearching HuggingFace for AAV2 dataset...")
try:
    api_url = "https://huggingface.co/api/datasets/bviggiano/aav2_capsid_viability"
    resp = urllib.request.urlopen(api_url)
    info = json.loads(resp.read())
    print(f"  Dataset info: {info.get('id', 'unknown')}")
    siblings = info.get("siblings", [])
    for s in siblings:
        print(f"  File: {s.get('rfilename')}")
except Exception as e:
    print(f"  HuggingFace API failed: {e}")

# Try direct parquet download with different paths
aav2_urls = [
    "https://huggingface.co/datasets/bviggiano/aav2_capsid_viability/resolve/main/data/train.parquet",
    "https://huggingface.co/datasets/bviggiano/aav2_capsid_viability/resolve/main/train-00000-of-00001.parquet",
    "https://huggingface.co/datasets/bviggiano/aav2_capsid_viability/resolve/main/aav2_capsid_viability.parquet",
]
for url in aav2_urls:
    try:
        print(f"\n  Trying: {url}")
        urllib.request.urlretrieve(url, f"{base}/aav2_capsid_viability.parquet")
        size = os.path.getsize(f"{base}/aav2_capsid_viability.parquet")
        print(f"  Downloaded: {size/1024/1024:.1f} MB")
        break
    except Exception as e:
        print(f"  Failed: {e}")

print("\nDone.")
