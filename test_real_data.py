import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_real_data_download():
    print("=" * 60)
    print("TEST: Real Data Download and Training Format Conversion")
    print("=" * 60)

    from pipeline.data_acquisition.real_data_loader import RealDataIntegrator

    integrator = RealDataIntegrator()

    print("\nDownloading real datasets...")
    try:
        real_data = integrator.load_real_data_for_training()

        print(f"\nLoaded data counts:")
        print(f"  AAV Tropism: {len(real_data['aav_tropism'])} samples")
        print(f"  LNP Delivery: {len(real_data['lnp_delivery'])} samples")
        print(f"  Immune Escape: {len(real_data['immune_escape'])} samples")

        if real_data["aav_tropism"]:
            sample = real_data["aav_tropism"][0]
            print(f"\nSample AAV tropism record:")
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Tissue: {sample.get('tissue', 'N/A')}")
            print(f"  Expression vector length: {len(sample.get('expression_vector', []))}")
            print(f"  Aging score: {sample.get('aging_score', 'N/A')}")

        if real_data["lnp_delivery"]:
            sample = real_data["lnp_delivery"][0]
            print(f"\nSample LNP delivery record:")
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Ionizable lipid: {sample.get('ionizable_lipid', 'N/A')}")
            print(f"  pKa: {sample.get('pka', 'N/A')}")

        saved_paths = integrator.save_real_training_data(real_data)
        print(f"\nSaved files:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("REAL DATA INTEGRATION TEST")
    print("=" * 60)

    result = test_real_data_download()

    print("\n" + "=" * 60)
    if result:
        print("REAL DATA INTEGRATION: PASS")
    else:
        print("REAL DATA INTEGRATION: FAIL")
        sys.exit(1)
