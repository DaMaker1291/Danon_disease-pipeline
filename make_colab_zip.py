import json, os, zipfile, random, shutil

random.seed(42)
data_dir = "C:/Users/supro/Downloads/life/data"
out_dir = os.path.join(data_dir, "colab_subsampled")
os.makedirs(out_dir, exist_ok=True)

print("Loading AAV...")
with open(os.path.join(data_dir, "real_screening_aav_tropism.json")) as f:
    aav = json.load(f)

print("Loading LNP...")
with open(os.path.join(data_dir, "real_screening_lnp_delivery.json")) as f:
    lnp = json.load(f)

print("Loading Immune...")
with open(os.path.join(data_dir, "real_screening_immune_escape.json")) as f:
    immune = json.load(f)

print(f"Full: AAV={len(aav)}, LNP={len(lnp)}, Immune={len(immune)}")

aav_sub = random.sample(aav, min(20000, len(aav)))
immune_sub = random.sample(immune, min(20000, len(immune)))

with open(os.path.join(out_dir, "real_screening_aav_tropism.json"), "w") as f:
    json.dump(aav_sub, f)
with open(os.path.join(out_dir, "real_screening_lnp_delivery.json"), "w") as f:
    json.dump(lnp, f)
with open(os.path.join(out_dir, "real_screening_immune_escape.json"), "w") as f:
    json.dump(immune_sub, f)

shutil.copy(os.path.join(data_dir, "LNPDB_real.csv"), out_dir)
shutil.copy(os.path.join(data_dir, "fit4function_screens.csv"), out_dir)

zip_path = os.path.join(data_dir, "..", "colab_data_small.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(out_dir):
        zf.write(os.path.join(out_dir, fname), fname)

zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Zip: {zip_size:.1f} MB")
print(f"AAV: {len(aav_sub)}, Immune: {len(immune_sub)}, LNP: {len(lnp)}")
