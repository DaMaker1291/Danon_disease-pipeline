import urllib.request, os, zipfile

base = "C:/Users/supro/Downloads/life/data"
zf = f"{base}/AlphaSeq.zip"

# Re-download
print("Re-downloading AlphaSeq...")
url = "https://zenodo.org/records/7783546/files/mit-ll%2FAlphaSeq_Antibody_Dataset-v2.0.0.zip?download=1"
urllib.request.urlretrieve(url, zf)
sz = os.path.getsize(zf) / 1024 / 1024
print(f"  Downloaded: {sz:.1f}MB")

# Extract outer zip
print("Extracting outer zip...")
with zipfile.ZipFile(zf, 'r') as z:
    for name in z.namelist():
        print(f"  {name}")
        z.extract(name, base)

# Now extract inner zips
print("\nExtracting inner CSV zips...")
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith(".csv.zip"):
            inner_zf = os.path.join(root, f)
            print(f"  Inner zip: {inner_zf}")
            with zipfile.ZipFile(inner_zf, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.csv'):
                        z.extract(name, base)
                        csv_path = os.path.join(base, os.path.basename(name))
                        print(f"    Extracted: {os.path.basename(name)} ({os.path.getsize(csv_path)/1024/1024:.1f}MB)")

# Cleanup
os.remove(zf)
print("\nDone.")
