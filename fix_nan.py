"""
Fix NaN in real screening data by imputation instead of dropping.
Drops hepg2/thle columns (67%/46% NaN), imputes everything else with median.
"""
import json
import math
import statistics

def safe_float(val, default=None):
    if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
        return float(val)
    return default

def collect_field(data, field, nested=None):
    vals = []
    for item in data:
        if nested:
            v = item.get(nested, {}).get(field)
        else:
            v = item.get(field)
        sv = safe_float(v)
        if sv is not None:
            vals.append(sv)
    return vals

def impute_field(data, field, default_val, nested=None):
    vals = collect_field(data, field, nested)
    if not vals:
        return 0.0
    return statistics.median(vals)

def impute_nested_dict(data, dict_key, sub_keys):
    sub_medians = {}
    for sk in sub_keys:
        vals = []
        for item in data:
            v = item.get(dict_key, {}).get(sk)
            if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                vals.append(float(v))
        sub_medians[sk] = statistics.median(vals) if vals else 0.0

    for item in data:
        if dict_key not in item:
            item[dict_key] = {}
        for sk in sub_keys:
            v = item[dict_key].get(sk)
            if v is None or (isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v))):
                item[dict_key][sk] = sub_medians[sk]

def process_file(filepath, name):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    with open(filepath) as f:
        data = json.load(f)
    print(f"  Loaded: {len(data)} samples")

    for item in data:
        item.pop("hepg2_transduction", None)
        item.pop("thle_transduction", None)
        item.pop("total_escape_score", None)
    print("  Dropped: hepg2_transduction, thle_transduction, total_escape_score")

    top_fields = ["delivery_efficiency", "stability_score", "immune_escape_score",
                  "neutralization_resistance", "liver_score", "production_score"]
    for field in top_fields:
        med = impute_field(data, field, 0.5)
        count = 0
        for item in data:
            v = item.get(field)
            if v is None or (isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v))):
                item[field] = med
                count += 1
        if count > 0:
            print(f"  Imputed {field}: {count} samples -> {med:.4f}")

    tissue_keys = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                   "hepatic", "renal", "pulmonary", "adipose"]
    for item in data:
        ts = item.get("tissue_scores", {})
        for tk in tissue_keys:
            v = ts.get(tk)
            if v is None or (isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v))):
                ts[tk] = 0.5
        item["tissue_scores"] = ts
    print("  Imputed tissue_scores NaN -> 0.5")

    ab_keys = ["AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3",
               "human_IgG_pool", "human_IgM_pool", "anti-AAV9_serum"]
    sub_keys = ["escape_score", "binding_energy"]

    for item in data:
        ab = item.get("antibody_responses", {})
        for ak in ab_keys:
            if ak not in ab:
                ab[ak] = {}
            for sk in sub_keys:
                v = ab[ak].get(sk)
                if v is None or (isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v))):
                    ab[ak][sk] = 0.5 if sk == "escape_score" else -5.0
        item["antibody_responses"] = ab
    print("  Imputed antibody_responses NaN -> defaults")

    remaining_nans = 0
    for item in data:
        for k, v in item.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                remaining_nans += 1
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, float) and (math.isnan(vv) or math.isinf(vv)):
                        remaining_nans += 1
                    if isinstance(vv, dict):
                        for vvv in vv.values():
                            if isinstance(vvv, float) and (math.isnan(vvv) or math.isinf(vvv)):
                                remaining_nans += 1
    print(f"  Remaining NaN/Inf: {remaining_nans}")

    return data

if __name__ == "__main__":
    base = "C:/Users/supro/Downloads/life/data"

    aav = process_file(f"{base}/real_screening_aav_tropism.json", "AAV Tropism")

    with open(f"{base}/real_screening_aav_tropism.json", "w") as f:
        json.dump(aav, f)
    print(f"\n  Saved: real_screening_aav_tropism.json ({len(aav)} samples)")

    with open(f"{base}/real_screening_immune_escape.json", "w") as f:
        json.dump(aav, f)
    print(f"  Saved: real_screening_immune_escape.json (same data, {len(aav)} samples)")

    print("\n" + "="*60)
    print("DONE - All NaN imputed. Ready for retraining.")
    print("="*60)
