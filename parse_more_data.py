import csv, json, os, math, zipfile

base = "C:/Users/supro/Downloads/life/data"
alphadir = f"{base}/mit-ll-AlphaSeq_Antibody_Dataset-a8f64a9"

# Extract nested AlphaSeq zips
print("Extracting AlphaSeq nested zips...")
for sub in ["antibody_dataset_1", "antibody_dataset_2"]:
    zf_path = None
    for f in os.listdir(os.path.join(alphadir, sub)):
        if f.endswith(".csv.zip"):
            zf_path = os.path.join(alphadir, sub, f)
            break
    if zf_path:
        with zipfile.ZipFile(zf_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    z.extract(name, base)
                    sz = os.path.getsize(os.path.join(base, name)) / 1024 / 1024
                    print(f"  Extracted: {name} ({sz:.1f}MB)")

# Parse AlphaSeq Dataset 1
print("\n" + "="*60)
print("AlphaSeq Dataset 1")
print("="*60)
try:
    csvs = [f for f in os.listdir(base) if f.startswith("MITLL") and f.endswith(".csv")]
    for csvf in csvs:
        with open(os.path.join(base, csvf), encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  {csvf}: {len(rows)} rows")
        print(f"  Columns: {list(rows[0].keys())[:15]}")
        for k in list(rows[0].keys())[:8]:
            vals = [r.get(k, "N/A") for r in rows[:3]]
            print(f"    {k}: {vals}")
except Exception as e:
    print(f"  Error: {e}")

# Parse LNPDB full - key columns for training
print("\n" + "="*60)
print("LNPDB Full - Key Columns")
print("="*60)
with open(f"{base}/LNPDB_full.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    lnpdb_rows = list(reader)
print(f"  Total rows: {len(lnpdb_rows)}")
# Find delivery/efficacy columns
all_cols = list(lnpdb_rows[0].keys())
efficacy_cols = [c for c in all_cols if any(x in c.lower() for x in ['efficacy', 'delivery', 'expression', 'silencing', 'knockdown', 'Activity', 'score', 'size', 'pdi', 'zeta'])]
print(f"  Efficacy columns: {efficacy_cols}")
for c in efficacy_cols[:10]:
    vals = [r[c] for r in lnpdb_rows[:5]]
    print(f"    {c}: {vals}")

# Parse Fit4Function in vitro
print("\n" + "="*60)
print("Fit4Function In Vitro")
print("="*60)
with open(f"{base}/fit4function_invitro.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    f4f_rows = list(reader)
print(f"  Total rows: {len(f4f_rows)}")
all_cols = list(f4f_rows[0].keys())
print(f"  All columns ({len(all_cols)}): {all_cols[:20]}")
seq_col = [c for c in all_cols if 'seq' in c.lower() or 'id' in c.lower()]
print(f"  Sequence/ID columns: {seq_col}")
binding_cols = [c for c in all_cols if 'b_' in c]
transduction_cols = [c for c in all_cols if '_tr_' in c]
print(f"  Binding columns ({len(binding_cols)}): {binding_cols[:5]}...")
print(f"  Transduction columns ({len(transduction_cols)}): {transduction_cols[:5]}...")
