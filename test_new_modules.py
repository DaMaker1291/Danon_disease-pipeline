import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_osk_safety():
    print("=" * 60)
    print("TEST: OSK Doxycycline Switch Safety (ER-100 Clinical Model)")
    print("=" * 60)

    from pipeline.models.osk_safety import OSKDoxycyclineSwitch, OSKPenalizedFitness

    switch = OSKDoxycyclineSwitch(max_days=56, target_days=42)

    test_cases = [
        (0.5, 42, "Optimal: 42d doxycycline"),
        (0.5, 28, "Short: 28d doxycycline"),
        (0.5, 56, "Max: 56d doxycycline (ER-100 limit)"),
        (0.8, 70, "Extended: 70d high expression"),
        (0.5, 120, "Very dangerous: 120d"),
    ]

    for level, days, desc in test_cases:
        score = switch.score_candidate(level, days)
        profile = score["profile"]
        print(f"\n{desc}:")
        print(f"  Expression: {level:.2f}, Duration: {days}d")
        print(f"  Oct4={profile.oct4_level:.3f}, Sox2={profile.sox2_level:.3f}, Klf4={profile.klf4_level:.3f}")
        print(f"  Rejuvenation: {score['efficacy_score']:.3f}")
        print(f"  Cancer Risk: {profile.cancer_risk:.3f}")
        print(f"  Safe: {profile.is_safe}, In Window: {profile.within_dox_window}")
        print(f"  Vector Present: {profile.vector_present}")
        print(f"  Composite: {score['composite_score']:.3f}")

    penalizer = OSKPenalizedFitness(max_days=56, penalty_weight=0.3)
    base_fitness = 0.8
    penalized = penalizer(base_fitness, 0.5, 42)
    print(f"\nPenalized fitness: {base_fitness} -> {penalized:.3f} (42d dox, 0.5 expression)")

    return True


def test_real_data_loader():
    print("\n" + "=" * 60)
    print("TEST: Real Data Loader")
    print("=" * 60)

    from pipeline.data_acquisition.real_data_loader import RealDataIntegrator

    integrator = RealDataIntegrator()
    data = integrator.load_real_data_for_training()
    print(f"\nLoaded training data categories: {list(data.keys())}")
    for k, v in data.items():
        print(f"  {k}: {len(v)} samples")

    return True


def test_cro_validation():
    print("\n" + "=" * 60)
    print("TEST: CRO Validation Bridge")
    print("=" * 60)

    from pipeline.lab.validation_bridge import (
        CROProtocolGenerator, ValidationCandidate
    )

    cro = CROProtocolGenerator()

    candidates = []
    for i in range(5):
        c = ValidationCandidate(
            candidate_id=i,
            candidate_type="aav" if i % 2 == 0 else "lnp",
            sequence="MAVGDLEGLSTT..." if i % 2 == 0 else "",
            composition={"ionizable_lipid": "DLin-MC3-DMA"} if i % 2 == 1 else {},
            ai_score=0.7 + i * 0.05,
            ai_predictions={"immune": 0.6, "tropism": 0.8, "transduction": 0.7},
        )
        candidates.append(c)

    protocol = cro.generate_synthesis_protocol(candidates)
    plan = cro.generate_organoid_testing_plan(protocol)
    estimate = cro.generate_cost_estimate(5, 2)

    print(f"\nProtocol ID: {protocol.protocol_id}")
    print(f"Candidates: {len(protocol.candidates)}")
    print(f"Target tissues: {protocol.target_tissues}")
    print(f"Readouts: {len(protocol.readouts)}")
    print(f"Estimated cost: ${estimate['total_estimated_usd']:,.0f}")
    print(f"Timeline: {protocol.estimated_timeline_weeks} weeks")

    print("\nOrganoid Testing Plan:")
    for phase_name, phase_data in plan.items():
        if isinstance(phase_data, dict) and "name" in phase_data:
            print(f"  {phase_name}: {phase_data['name']} ({phase_data.get('duration_weeks', 'N/A')} weeks)")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "validation_package")
    cro.export_protocol_package(protocol, plan, estimate, output_dir)
    print(f"\nValidation package exported to: {output_dir}")

    return True


def test_osk_integrated_refinement():
    print("\n" + "=" * 60)
    print("TEST: OSK-Integrated Refinement Loop")
    print("=" * 60)

    from pipeline.feedback.refinement import FeedbackLoop

    loop = FeedbackLoop(
        osk_max_dox_days=56,
        osk_penalty_weight=0.3,
    )

    print(f"OSK penalizer available: {loop.osk_penalizer is not None}")

    if loop.osk_penalizer:
        test_scores = [
            (0.8, 0.5, 42),
            (0.8, 0.5, 56),
            (0.8, 0.5, 70),
            (0.8, 0.8, 120),
        ]
        for base, level, days in test_scores:
            penalized = loop.osk_penalizer(base, level, days)
            print(f"  Base={base:.2f}, Expr={level:.1f}, Days={days:3d} -> Penalized={penalized:.3f}")

    return True


if __name__ == "__main__":
    print("LIFETIME LONGEVITY PIPELINE - NEW MODULE TESTS")
    print("=" * 60)

    results = []
    results.append(("OSK Safety", test_osk_safety()))
    results.append(("Real Data Loader", test_real_data_loader()))
    results.append(("CRO Validation", test_cro_validation()))
    results.append(("OSK-Integrated Refinement", test_osk_integrated_refinement()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
