import csv, json, os, math

base = "C:/Users/supro/Downloads/life/data"

# 1. Parse LNPDB_full.csv
print("="*60)
print("LNPDB Full")
print("="*60)
try:
    with open(f"{base}/LNPDB_full.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())[:15]}...")
    for k in list(rows[0].keys())[:20]:
        vals = [r[k] for r in rows[:5]]
        print(f"    {k}: {vals}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Parse LNP_Atlas.csv
print("\n" + "="*60)
print("LNP Atlas")
print("="*60)
try:
    with open(f"{base}/LNP_Atlas.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())[:15]}...")
    for k in list(rows[0].keys())[:20]:
        vals = [r[k] for r in rows[:5]]
        print(f"    {k}: {vals}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Check fit4function_invitro.csv
print("\n" + "="*60)
print("Fit4Function In Vitro")
print("="*60)
try:
    with open(f"{base}/fit4function_invitro.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())[:15]}...")
    for k in list(rows[0].keys())[:20]:
        vals = [r[k] for r in rows[:5]]
        print(f"    {k}: {vals}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Try AlphaSeq from alternative source
print("\n" + "="*60)
print("AlphaSeq")
print("="*60)
# Delete corrupted zip
zf = f"{base}/AlphaSeq.zip"
if os.path.exists(zf):
    os.remove(zf)
    print("  Removed corrupted zip")

# Try direct download of just the CSV
url = "https://zenodo.org/records/7783546/files/mit-ll%2FAlphaSeq_Antibody_Dataset-v2.0.0.zip?download=1"
print(f"  Attempting re-download...")
try:
    import urllib.request
    urllib.request.urlretrieve(url, zf)
    print(f"  Downloaded: {os.path.getsize(zf)/1024/1024:.1f}MB")
except Exception as e:
    print(f"  Download failed: {e}")
