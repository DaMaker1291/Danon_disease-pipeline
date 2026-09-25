"""Regenerate outputs_real/clinical_twin.txt from the live census and reports.

Usage: python _rebuild_clinical_twin_report.py

The file is regenerated rather than hand-edited so its refusal counts cannot
drift from the census the claims are checked against. It had drifted once
already: the stream-4 line still said "no sequential scan pairs exist" after
the census started reporting the manifest's 168 patients with a usable pair.
"""

from __future__ import annotations

import textwrap

from danon.clinical_twin import clinical_stream_census, stream3_report

OUT = "outputs_real/clinical_twin.txt"
WIDTH = 100


def main() -> int:
    census = clinical_stream_census()
    s3 = stream3_report()
    s4 = census["stream4_sequential"]

    lines = [
        "EVOTWIN-10X: CLINICAL DATASTREAM STATUS (four routine hospital tests)",
        "=" * 72,
        "",
        "Per-stream census (live counts from data/):",
    ]
    for key, entry in census.items():
        detail = entry.get("detail", "")
        lines.append(
            "  {:<22} {:<11} {}".format(
                key, entry.get("status", "?"), textwrap.shorten(detail, WIDTH - 36)
            )
        )

    lines += [
        "",
        "STREAM 3 MEASURED (6 BraTS cases, t1ce -> vasculature, paired same-seed):",
        "  MTD PFS paired delta (real - null), {} runs: mean {:.2f} d "
        "(sd {:.2f}, max {:.0f})".format(
            s3["n_paired_runs_per_arm"],
            s3["mtd_pfs_delta_mean_days"],
            s3["mtd_pfs_delta_sd_days"],
            s3["mtd_pfs_delta_max_abs_days"],
        ),
        "  Steering: horizon-censored in BOTH geometries (paired delta exactly 0)",
        "",
        "  case    MTD real/null   mean enhancement",
    ]
    for case, c in s3["cases"].items():
        lines.append(
            "  {:<8} {:>4} / {:>5}      {:.3f}".format(
                case, int(c["mtd_pfs_median_real"]), int(c["mtd_pfs_median_null"]),
                c["mean_enhancement_in_tumour"],
            )
        )

    lines += [
        "",
        "HONEST READING: at engine scale the vessel-source variation contributes",
        "~1 day of PFS; the sanctuary structure lives in D(x) (tissue classes),",
        "which both geometries share. t1ce is a vascular PROXY, not measured Ktrans.",
        "",
        "STREAM 4 RESOLUTION (a DESIGN calculation, not a measurement):",
    ]
    res = s4.get("resolution", {})
    r = res.get("resolution")
    if r:
        lines += [
            "  manifest cohort: {} patients with a usable T1-Gd pair ({} pairs)".format(
                res["manifest_n_patients_with_usable_pair"], res["manifest_n_usable_pairs"]
            ),
            "  reference cohort: n={} scored at C={:.4f} +/- {:.4f} (OS)".format(
                res["source_n_patients"], res["source_base_c"], res["source_base_fold_sd"]
            ),
            "  over {} resampled {}-patient cohorts: C={:.4f} +/- {:.4f}, mean fold "
            "sd {:.4f}".format(
                r["n_draws"], r["n_patients"], r["mean_c"], r["se_across_draws"],
                r["mean_fold_sd"],
            ),
            "  => a cohort this size resolves |delta_C| > {:.4f} (1 SE); fold "
            "yardstick {:.4f}".format(r["floor_optimistic"], r["floor_conservative"]),
            "  Stream 4 is an OBSERVATION: it calibrates the engine's derived velocity",
            "  and cannot move a C-index until a hazard mapping is declared. The images",
            "  are not downloaded, so nothing here measured perfusion, fluorescence,",
            "  Ki-67 or delta-volume.",
        ]
    else:
        lines.append(
            "  not computed: {}".format(res.get("error", "design calculation unreadable"))
        )

    with open(OUT, "w", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
