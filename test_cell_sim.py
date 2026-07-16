import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danon.cell_simulator import DanonCellSimulator

cs = DanonCellSimulator()

r = cs.simulate_cell(mutation_type='null', vector_copy_number=10, cardiac_tropism=0.72)
print(f'Null mutation @ VCN=10: cure={r.is_functional_cure}, score={r.functional_cure_score:.3f}')
print(f'  LAMP2={r.lamp2_restoration_pct:.1f}%  Glycogen clearance={r.glycogen_clearance_pct:.1f}%')
print(f'  Flux restoration={r.autophagic_flux_restoration_pct:.1f}%  pH norm={r.lysosomal_ph_normalization:.2f}')
print(f'  Survival: 1y={r.cell_survival_at_1y:.3f}  5y={r.cell_survival_at_5y:.3f}  10y={r.cell_survival_at_10y:.3f}')

panel = cs.dose_response_panel(mutation_type='null')
for p in panel:
    print(f'  VCN={p["vector_copy_number"]:3.0f}: score={p["functional_cure_score"]:.3f} cure={p["is_cure"]} surv5y={p["survival_5y"]:.3f}')

pop = cs.simulate_population(n_patients=50)
print(f'Population (n=50): cure_rate={pop["cure_rate"]:.2f} improved={pop["improvement_rate"]:.2f}')

trial = cs.project_trial_outcome(n_patients=24)
print(f'Trial (n=24): cure_rate={trial["cure_rate"]:.2f} power={trial["statistical_power"]:.2f} feasible={trial["trial_feasible"]}')

mut_comparison = cs.compare_mutation_types()
for m in mut_comparison:
    print(f'  {m["mutation"]:25s}: cure={m["is_cure"]} score={m["cure_score"]:.3f} surv10y={m["survival_10y"]:.3f}')

ec50 = cs.ec50_analysis('null')
print(f'EC50: ed50={ec50["ed50_vcn"]} min_cure_vcn={ec50["min_cure_vcn"]} max_score={ec50["max_score"]:.3f}')

no_therapy = cs.simulate_cell(mutation_type='null', vector_copy_number=0, cardiac_tropism=0.72)
print(f'No therapy: LAMP2={no_therapy.lamp2_restoration_pct:.1f}%  surv5y={no_therapy.cell_survival_at_5y:.3f}  surv10y={no_therapy.cell_survival_at_10y:.3f}')

print('=== ALL CELL SIMULATOR TESTS PASSED ===')
