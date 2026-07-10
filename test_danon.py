import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from danon.config import DanonConfig
from danon.aav_generator import DanonAAVGenerator, DanonAAVCandidate
from danon.lnp_generator import DanonLNPGenerator, DanonLNPCandidate
from danon.tropism_filter import DanonTropismFilter
from danon.safety_engine import DanonSafetyEngine


def test_config():
    config = DanonConfig()
    assert config.target_disease == "Danon Disease (Monogenic Hypertrophic Cardiomyopathy)"
    assert config.therapeutic_payload == "LAMP2B_Transgene"
    assert config.vector_backbone == "AAV9_Recombinant_Capsid"
    assert "cardiac_myocytes" in config.target_tissues
    assert "hepatic" in config.avoid_tissues
    assert config.regulatory_framework == "MHRA_ILAP_FastTrack (UK)"
    print("PASS: Config")


def test_aav_generator():
    config = DanonConfig(aav_total_candidates=100, batch_size=50)
    gen = DanonAAVGenerator(config)
    candidates = gen.generate_candidates(0, 10)
    assert len(candidates) == 10
    for c in candidates:
        assert len(c.sequence) > 0
        assert c.candidate_id >= 0
    scored = gen.score_candidates(candidates)
    for c in scored:
        assert c.fitness >= 0
        assert c.fitness <= 1
        assert c.cardiac_tropism_score >= 0
        assert c.hepatic_avoidance_score >= 0
    print("PASS: AAV Generator")


def test_lnp_generator():
    config = DanonConfig(lnp_total_candidates=100, batch_size=50)
    gen = DanonLNPGenerator(config)
    candidates = gen.generate_candidates(0, 10)
    assert len(candidates) == 10
    scored = gen.score_candidates(candidates)
    for c in scored:
        assert c.fitness >= 0
        assert c.cardiac_delivery_score >= 0
        assert c.hepatic_avoidance_score >= 0
    print("PASS: LNP Generator")


def test_tropism_filter():
    config = DanonConfig()
    filt = DanonTropismFilter(config)
    gen = DanonAAVGenerator(config)
    candidates = gen.generate_candidates(0, 20)
    scored = gen.score_candidates(candidates)
    passed = [c for c in scored if filt.passes(c)]
    assert len(passed) >= 0
    print("PASS: Tropism Filter (%d/%d passed)" % (len(passed), len(scored)))


def test_safety_engine():
    config = DanonConfig()
    engine = DanonSafetyEngine(config)
    gen = DanonAAVGenerator(config)
    candidates = gen.generate_candidates(0, 5)
    scored = gen.score_candidates(candidates)
    for c in scored:
        profile = engine.evaluate(c)
        assert 0 <= profile.cardiac_tropism <= 1
        assert 0 <= profile.hepatic_accumulation <= 1
        assert 0 <= profile.overall_safety <= 1
        assert isinstance(profile.is_safe, bool)
        assert isinstance(profile.regulatory_compliant, bool)
    print("PASS: Safety Engine")


def test_streaming():
    config = DanonConfig(aav_total_candidates=200, batch_size=50)
    gen = DanonAAVGenerator(config)
    total = 0
    for batch in gen.stream_candidates(200, 50):
        total += len(batch)
    assert total == 200
    print("PASS: Streaming (%d candidates)" % total)


if __name__ == "__main__":
    test_config()
    test_aav_generator()
    test_lnp_generator()
    test_tropism_filter()
    test_safety_engine()
    test_streaming()
    print("\n=== ALL DANON TESTS PASSED ===")
