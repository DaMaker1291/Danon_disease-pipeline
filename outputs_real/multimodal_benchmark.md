# Retrospective multimodal benchmark

## Data sources probed
- BraTS 2021 expert segmentations (local) [LOCAL] cases=6, n_segmentation_files=6, timepoints_per_case=1
    supports: real pre-operative tumour geometry and a label-derived proliferative map for the engine grid
    cannot: a follow-up timepoint: the release is pre-operative only
    cannot: vasculature: no perfusion map is distributed with the static release
- ChEMBL measured activity records (local cache) [LOCAL] n_activity_files=1, n_activity_records_in_first=166, n_molecule_caches=1
    supports: measured potency with units, relation and target preserved, and curated molecular descriptors
    cannot: in-vivo efficacy: 2D monolayer IC50 is not 3D tumour kill
- GDC TCGA-GBM clinical export (local) [LOCAL] n_cases=617, path=data/gdc_clinical/tcga_gbm_cases.json
    supports: real recorded progression and overall-survival endpoints for a real patient cohort, used as the external anchor and holdout
    cannot: patient-level linkage to the imaging cohort: BraTS and GDC anonymise differently
- GDC STAR gene-count matrices (local) [LOCAL] assay=bulk RNA-seq STAR counts, n_expression_files=6
    supports: bulk transcriptomes that parameterise a sub-clone ratio by calibrated inference
    cannot: single-cell resolution: bulk RNA-seq cannot resolve individual sub-clones (marked PROVEN_FAILED in danon.genomics_intake)
- TCIA collection BraTS [RETIRED] collection=BraTS
    cannot: any imaging-based test, including the barcode-based link to a GDC clinical export
- TCIA collection ICDC-Glioma [ABSENT] collection=ICDC-Glioma, modalities=['MR'], n_patients=57, n_seg_series=0, n_series=650, patients_checked_for_timepoints=2, patients_with_multi_timepoint=0
    supports: single-timepoint anonymised MRI with a documented collection page
    cannot: an expert tumour mask to score a predicted field against
    cannot: a second timepoint, so there is no follow-up scan to compare to
- TCIA NBIA v1 API [AVAILABLE] glioma_candidates_absent=['BraTS', 'TCGA-GBM', 'TCGA-LGG'], glioma_candidates_present=['ICDC-Glioma', 'UPENN-GBM'], n_collections=156
    supports: anonymised imaging for collections that are still served
    cannot: patient-level joining of radiology to an external clinical export when the collection that carried the shared identifier has been withdrawn
- TCIA collection TCGA-GBM [RETIRED] collection=TCGA-GBM
    cannot: any imaging-based test, including the barcode-based link to a GDC clinical export
- TCIA collection TCGA-LGG [RETIRED] collection=TCGA-LGG
    cannot: any imaging-based test, including the barcode-based link to a GDC clinical export
- TCIA collection UPENN-GBM [ABSENT] collection=UPENN-GBM, min_followup_interval_days=28, modalities=['MR'], n_dsc_series=490, n_longitudinal_with_perfusion=41, n_patients=630, n_patients_dsc_perfusion=456, n_patients_processed_perfusion=628, n_processed_series=2607, n_seg_series=0, n_series=3680, patients_checked_for_timepoints=630, patients_multi_date_any_interval=43, patients_with_multi_timepoint=41, sampled_patients_checked=2, sampled_patients_with_multi_timepoint=0, timepoint_scope=timepoints scored over every patient in the collection-wide series listing, not a sample, timepoint_source_file=data/tcia_probe/upenn_series.json
    supports: anonymised glioma MRI with a documented collection page
    supports: a follow-up timepoint for 41 of 630 patients (>=28 day interval), each with DSC perfusion series
    supports: DSC perfusion acquisition for 456 patients and CaPTk-processed maps for 628 (rCBV/Ktrans proxies for stream 3)
    cannot: an expert tumour mask to score a predicted field against

## Test battery
- [SUPPORTED] test1_discrimination: does the model rank real patients by real outcome?
      measured by: danon.predictive.kfold_holdout_c_index / harrell_c_index
      basis: held-out Cox discrimination on the real GDC TCGA-GBM clinical export, with the fit and the score on separate patient splits
      blocker: the engine itself has no per-patient prediction to rank (danon.predictive.engine_discrimination_verdict), so the C-index reported is for a covariate model, not for the scheduler
- [REFUSED] test2_spatial_dice: does the predicted spatial field match a real follow-up scan?
      measured by: danon.imaging (geometry intake) -- no Dice estimator wired
      basis: no reachable glioma imaging collection carries both an expert tumour mask and a second timepoint for the same patient
      blocker: a collection with DICOM SEG (or equivalent mask series) on glioma MRI; at least one patient with two imaging timepoints, so a predicted day-90 field can be compared to an observed recurrence scan; an explicit mapping between that patient and an administered schedule, without which a mismatch cannot be attributed to the model
- [SUPPORTED] test3_paired_control: does the recommended schedule beat standard care on identical geometry?
      measured by: danon.outcome_export.export_outcome_cards / danon.campaign.run_campaign
      basis: both arms run on the same real BraTS-derived geometry with replicates, reporting the burden and progression delta paired per geometry rather than across cohorts
      blocker: the observed cohort and the imaging cohort are different people, so the size of the advantage is a model result, not a treatment effect measured in patients

## Measured outcomes (tests 1 and 3)

### Test 1 -- held-out discrimination on the real GDC cohort
```
os: os: C = 0.656 +/- 0.025 over 5 folds (age alone 0.653) -- real but weak discrimination on held-out patients, consistent with the published prognostic value of age and diagnosis in GBM
    n=595 patients; 595 patients with a usable os endpoint; 491 events, 104 censored at last contact; 1 patient(s) recorded Dead with no days_to_death: censored at last contact because there is no event time; 21 dropped: no age recorded; 1 dropped: no usable time or endpoint
progression: progression: C = 0.534 +/- 0.053 over 5 folds (age alone 0.539) -- no usable held-out discrimination from these covariates
    n=594 patients; 594 patients with a usable progression endpoint; 372 events, 222 censored at last contact; 21 dropped: no age recorded; 2 dropped: no usable time or endpoint
```

### Test 3 -- paired control on real BraTS-derived geometry
paired control over 9 card(s): progression-free 9/9 on the policy vs 3/9 on MTD; median PFS extension +0.267 (range +0.000 to +0.342), censored in 6 where the policy never progressed inside the horizon; median toxicity reduction 0.013
  - 6 card(s) have the MTD arm progressing while the policy arm does not reach threshold inside the 365-day horizon, so their PFS extension is a lower bound rather than a measured survival gain
  - both arms are the same model, so this is the model's own contrast on real geometry, not a clinical benefit measured in patients

## Spatial verdict
spatial Dice test REFUSED: no reachable glioma imaging collection carries both an expert tumour mask and a second timepoint for the same patient
  - TCIA collection TCGA-GBM: RETIRED -- not served by the API (TCGA-GBM)
  - TCIA collection TCGA-LGG: RETIRED -- not served by the API (TCGA-LGG)
  - TCIA collection UPENN-GBM: 630 patient(s), 3680 series, modalities ['MR'], 0 segmentation series, 41/630 patient(s) with more than one StudyDate (timepoints scored over every patient in the collection-wide series listing, not a sample)
  - TCIA collection BraTS: RETIRED -- not served by the API (BraTS)
  - TCIA collection ICDC-Glioma: 57 patient(s), 650 series, modalities ['MR'], 0 segmentation series, 0/2 patient(s) with more than one StudyDate (sampled)
  needs: a collection with DICOM SEG (or equivalent mask series) on glioma MRI
  needs: at least one patient with two imaging timepoints, so a predicted day-90 field can be compared to an observed recurrence scan
  needs: an explicit mapping between that patient and an administered schedule, without which a mismatch cannot be attributed to the model

## Limits
- Data presence is not model validity. A supported test says an estimate could be computed; it does not say the estimate is good.
- The spatial test is refused because no reachable collection carries both masks and a second timepoint -- that is a statement about public data availability, not evidence that the engine's spatial predictions are wrong. The distinction matters: absence of a test is not a negative result.
- The imaging cohort (BraTS-derived) and the outcome cohort (GDC TCGA-GBM) anonymise differently and cannot be joined at patient level. Every comparison across them is between distributions, never between matched individuals.
- The paired-control delta is produced by both arms of the same model. It measures the model's own contrast, not a clinical benefit in patients, and cannot be reported as a survival gain.
- Reachability was probed from one environment on one date; a collection added later can change the second test's verdict, which is why the verdict is computed from recorded counts rather than asserted.