import os, zipfile

base = "C:/Users/supro/Downloads/life/data"
alphadir = f"{base}/mit-ll-AlphaSeq_Antibody_Dataset-a8f64a9"

# Extract dataset 2
print("Extracting dataset 2...")
inner_zf = f"{alphadir}/antibody_dataset_2/MITLL_AAlphaBio_Ab_Binding_dataset2.csv.zip"
with zipfile.ZipFile(inner_zf, 'r') as z:
    for name in z.namelist():
        if name.endswith('.csv') and not name.startswith('._'):
            z.extract(name, base)
            csv_path = os.path.join(base, os.path.basename(name))
            print(f"  Extracted: {os.path.basename(name)} ({os.path.getsize(csv_path)/1024/1024:.1f}MB)")

# List all large CSVs
print("\nAll data files:")
for f in sorted(os.listdir(base)):
    fp = os.path.join(base, f)
    if os.path.isfile(fp) and not f.startswith('.'):
        sz = os.path.getsize(fp) / 1024 / 1024
        if sz > 0.5:
            print(f"  {f}: {sz:.1f}MB")
