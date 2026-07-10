"""
Parse new datasets into training JSON format.
1. LNPDB full (19K formulations) -> real_screening_lnp_delivery.json
2. Fit4Function in vitro + screens -> augment AAV training data
3. AlphaSeq antibody (sample) -> augment immune escape training
"""
import csv, json, os, math, re, random

base = "C:/Users/supro/Downloads/life/data"

# =========================================================================
# 1. Parse LNPDB Full -> LNP training format
# =========================================================================
print("="*60)
print("1. Parsing LNPDB Full -> LNP delivery training data")
print("="*60)

lipid_map = {
    "DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2, "DODAP": 3,
    "DLin-DMA": 4, "cKK-E11": 5, "None": 0,
}
peg_map = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
helper_map = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3, "None": 0, "": 0}

def parse_tail_from_smiles(smiles):
    if not smiles or smiles == "NA" or smiles == "None":
        return 16, 2
    c_count = smiles.count("C") + smiles.count("c")
    double_bonds = smiles.count("=")
    tail = max(12, min(22, c_count // 2))
    unsat = max(0, min(5, double_bonds))
    return tail, unsat

def safe_float(val, default=0.0):
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default

lnp_training = []
with open(f"{base}/LNPDB_full.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    seen = set()
    for row in reader:
        form_id = row.get("Formulation_ID", "")
        if form_id in seen:
            continue
        seen.add(form_id)

        il_name = row.get("IL_name", "None")
        hl_name = row.get("HL_name", "None") or "None"
        chl_name = row.get("CHL_name", "Cholesterol")
        peg_name = row.get("PEG_name", "DMG-PEG2000")

        il_mol = safe_float(row.get("IL_molratio", 42))
        hl_mol = safe_float(row.get("HL_molratio", 0))
        chl_mol = safe_float(row.get("CHL_molratio", 48))
        peg_mol = safe_float(row.get("PEG_molratio", 10))
        total = il_mol + hl_mol + chl_mol + peg_mol
        if total == 0:
            total = 100

        tail, unsat = parse_tail_from_smiles(row.get("IL_SMILES", ""))
        exp_val = safe_float(row.get("Experiment_value", 0.5))
        model_target = row.get("Model_target", "in_vitro")
        exp_method = row.get("Experiment_method", "luminescence_normalized")

        if "hemolysis" in exp_method.lower():
            cytotox = exp_val / 100.0 if exp_val > 1 else exp_val
            delivery = 0.5
        else:
            delivery = max(0.0, min(1.0, exp_val / 10.0)) if exp_val > 0 else 0.5
            cytotox = 0.1

        lnp_training.append({
            "ionizable_lipid": il_name,
            "helper_lipid": hl_name,
            "peg_lipid": peg_name,
            "ionizable_frac": round(il_mol / total, 4),
            "peg_frac": round(peg_mol / total, 4),
            "cholesterol_frac": round(chl_mol / total, 4),
            "pka": 6.4,
            "tail_length": tail,
            "unsaturation": unsat,
            "delivery_efficiency": round(delivery, 4),
            "immune_activation": 0.2,
            "cytotoxicity": round(cytotox, 4),
            "organoid_barcode_counts": {"heart_organoid": 500, "brain_organoid": 500, "liver_organoid": 500, "joint_organoid": 500},
            "source": "LNPDB_full",
            "model_target": model_target,
        })

# Merge with existing LNPDB_real (only add LNPDB_full samples not already present)
with open(f"{base}/real_screening_lnp_delivery.json") as f:
    existing = json.load(f)
existing_lnpdb_full = [x for x in existing if x.get("source") == "LNPDB_full"]
existing_other = [x for x in existing if x.get("source") != "LNPDB_full"]
print(f"  LNPDB_full new: {len(lnp_training)} unique formulations")
print(f"  LNPDB_full existing: {len(existing_lnpdb_full)} (will skip)")
print(f"  Other existing: {len(existing_other)}")
all_lnp = existing_other + lnp_training
print(f"  Combined: {len(all_lnp)} total")

tmp_path = f"{base}/real_screening_lnp_delivery.json.tmp"
with open(tmp_path, "w") as f:
    json.dump(all_lnp, f)
os.replace(tmp_path, f"{base}/real_screening_lnp_delivery.json")
print(f"  Saved: real_screening_lnp_delivery.json ({len(all_lnp)} samples)")

# =========================================================================
# 2. Parse Fit4Function in vitro -> augment AAV training
# =========================================================================
print("\n" + "="*60)
print("2. Parsing Fit4Function in vitro + screens -> AAV augmentation")
print("="*60)

with open(f"{base}/fit4function_screens.csv", encoding="utf-8") as f:
    screens = list(csv.DictReader(f))
print(f"  Screens: {len(screens)} rows")

# Build sequence lookup: row index -> sequence
screen_seqs = {}
for i, row in enumerate(screens):
    seq = row.get("Sequence", row.get("sequence", ""))
    if seq and len(seq) >= 3:
        screen_seqs[i] = seq

with open(f"{base}/fit4function_invitro.csv", encoding="utf-8") as f:
    invitro = list(csv.DictReader(f))
print(f"  In vitro: {len(invitro)} rows")

# Merge: invitro row i -> screen row i for sequence
invitro_with_seq = []
for i, row in enumerate(invitro):
    if i in screen_seqs:
        seq = screen_seqs[i]
        binding_cols = [c for c in row if "_b_" in c]
        tr_cols = [c for c in row if "_tr_" in c]

        binding_vals = [safe_float(row[c]) for c in binding_cols if safe_float(row[c], None) is not None]
        tr_vals = [safe_float(row[c]) for c in tr_cols if safe_float(row[c], None) is not None]

        avg_binding = sum(binding_vals) / len(binding_vals) if binding_vals else 0.5
        avg_transduction = sum(tr_vals) / len(tr_vals) if tr_vals else 0.5

        # Normalize to 0-1
        delivery = min(1.0, max(0.0, avg_transduction / 20.0))
        stability = min(1.0, max(0.0, avg_binding / 10.0))

        tissues = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                   "hepatic", "renal", "pulmonary", "adipose"]
        tissue_scores = {t: 0.5 for t in tissues}
        tissue_scores["neuronal"] = min(1.0, max(0.0, safe_float(row.get("hCMECd3_b_1", 0)) / 5.0))
        tissue_scores["hepatic"] = min(1.0, max(0.0, delivery))

        invitro_with_seq.append({
            "sequence": seq[:50],
            "tissue_scores": tissue_scores,
            "tropism_target": "hepatic",
            "delivery_efficiency": round(delivery, 4),
            "immune_escape_score": 0.43,
            "stability_score": round(stability, 4),
            "source": "Fit4Function_invitro",
            "mutations": [],
            "production_score": -0.7,
            "liver_score": -0.9,
            "antibody_responses": {},
            "total_escape_score": 0.43,
            "neutralization_resistance": 0.63,
        })

print(f"  Matched invitro+screen: {len(invitro_with_seq)} samples")

# Load existing AAV data (only keep non-invitra samples to avoid double-counting)
with open(f"{base}/real_screening_aav_tropism.json") as f:
    existing_aav = json.load(f)
existing_aav_invitro = [x for x in existing_aav if x.get("source") == "Fit4Function_invitro"]
existing_aav_other = [x for x in existing_aav if x.get("source") != "Fit4Function_invitro"]
all_aav = existing_aav_other + invitro_with_seq
print(f"  Fit4Function invitro new: {len(invitro_with_seq)}")
print(f"  Fit4Function invitro existing: {len(existing_aav_invitro)} (will skip)")
print(f"  Other existing AAV: {len(existing_aav_other)}")
print(f"  Combined: {len(all_aav)} total")

tmp1 = f"{base}/real_screening_aav_tropism.json.tmp"
with open(tmp1, "w") as f:
    json.dump(all_aav, f)
os.replace(tmp1, f"{base}/real_screening_aav_tropism.json")
tmp2 = f"{base}/real_screening_immune_escape.json.tmp"
with open(tmp2, "w") as f:
    json.dump(all_aav, f)
os.replace(tmp2, f"{base}/real_screening_immune_escape.json")
print(f"  Saved: aav + immune JSONs ({len(all_aav)} samples)")

# =========================================================================
# 3. Sample AlphaSeq -> immune escape augmentation
# =========================================================================
print("\n" + "="*60)
print("3. Sampling AlphaSeq -> immune escape augmentation")
print("="*60)

alpha_rows = []
for csvf in ["MITLL_AAlphaBio_Ab_Binding_dataset.csv", "MITLL_AAlphaBio_Ab_Binding_dataset2.csv"]:
    fpath = f"{base}/{csvf}"
    if not os.path.exists(fpath):
        continue
    print(f"  Reading {csvf}...")
    with open(fpath, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            seq = row.get("Sequence", "")
            aff = row.get("Pred_affinity", "")
            if not seq or not aff:
                continue
            try:
                aff_val = float(aff)
                if math.isnan(aff_val) or math.isinf(aff_val):
                    continue
            except ValueError:
                continue

            # Normalize affinity to escape score (high affinity = low escape)
            escape = max(0.0, min(1.0, 1.0 - (aff_val + 0.5) / 3.5))

            alpha_rows.append({
                "sequence": seq[:50],
                "escape_scores": [escape] * 7,
                "binding_energies": [aff_val] * 7,
                "total_escape": round(escape, 4),
                "neutralization_resistance": round(min(1.0, max(0.0, aff_val / 3.0)), 4),
                "source": "AlphaSeq",
                "antibody_responses": {
                    "AAV2_Ab4": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "AAV2_Ab58": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "AAV8_Ab1": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "AAV9_Ab3": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "human_IgG_pool": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "human_IgM_pool": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                    "anti-AAV9_serum": {"escape_score": round(escape, 4), "binding_energy": round(aff_val, 4)},
                },
            })
            count += 1
            if count >= 100000:
                break
    print(f"    Sampled: {count}")

print(f"  AlphaSeq samples: {len(alpha_rows)}")

# Combine with existing (keep only non-AlphaSeq samples to avoid double-counting)
with open(f"{base}/real_screening_immune_escape.json") as f:
    existing_immune = json.load(f)
existing_alpha = [x for x in existing_immune if x.get("source") == "AlphaSeq"]
existing_immune_other = [x for x in existing_immune if x.get("source") != "AlphaSeq"]

# existing_immune_other is same as AAV data, keep it
all_immune = existing_immune_other + alpha_rows
print(f"  AlphaSeq new: {len(alpha_rows)}")
print(f"  AlphaSeq existing: {len(existing_alpha)} (will skip)")
print(f"  Other existing immune: {len(existing_immune_other)}")
print(f"  Combined: {len(all_immune)} total")

tmp3 = f"{base}/real_screening_immune_escape.json.tmp"
with open(tmp3, "w") as f:
    json.dump(all_immune, f)
os.replace(tmp3, f"{base}/real_screening_immune_escape.json")
print(f"  Saved: real_screening_immune_escape.json ({len(all_immune)} samples)")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  AAV tropism: {len(all_aav)} samples")
print(f"  LNP delivery: {len(all_lnp)} samples")
print(f"  Immune escape: {len(all_immune)} samples")
print("  All data parsed and saved.")
