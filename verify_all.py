import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.data_acquisition.real_screening_loader import RealScreeningDataLoader
loader = RealScreeningDataLoader()
data = loader.load_all_real_screening_data()
print("LNPDB (cached):", len(data['lnp_delivery']), "real formulations")
print("All keys present:", all(k in data for k in ['aav_tropism','lnp_delivery','immune_escape']))

from pipeline.models.osk_safety import OSKDoxycyclineSwitch
s = OSKDoxycyclineSwitch(max_days=56)
score = s.score_candidate(0.5, 42)
p = score['profile']
print("OSK ER-100 Model:")
print("  42d dox: composite=%.3f, cancer_risk=%.3f, safe=%s" % (score['composite_score'], p.cancer_risk, p.is_safe))
score2 = s.score_candidate(0.8, 70)
p2 = score2['profile']
print("  70d dox: composite=%.3f, cancer_risk=%.3f, safe=%s" % (score2['composite_score'], p2.cancer_risk, p2.is_safe))

print("Validation package:", os.path.exists(os.path.join('data', 'validation_package', 'cro_protocol.json')))
print("Abstract:", os.path.exists('abstract.md'))
print("Cache dir:", os.path.exists('data/_remote_cache'))
for f in os.listdir('data/_remote_cache'):
    print("  ", f)
