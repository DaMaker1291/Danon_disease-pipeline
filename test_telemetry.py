import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_telemetry_logger():
    print("=" * 60)
    print("TEST: PrecisionTelemetryLogger")
    print("=" * 60)

    from pipeline.training.telemetry import PrecisionTelemetryLogger
    import numpy as np

    output_dir = os.path.join(os.path.dirname(__file__), "..", "diagnostics", "test")
    logger = PrecisionTelemetryLogger(output_dir)

    for i in range(50):
        train_loss = 1.0 / (i + 1) + np.random.normal(0, 0.05)
        val_loss = 1.0 / (i + 1) + np.random.normal(0, 0.08)
        lr = 3e-4 * (1 - i / 50)
        grad_norm = np.random.uniform(0.1, 1.0)
        logger.log_step(train_loss, val_loss, lr, grad_norm)

    print("Generating loss plot...")
    path = logger.generate_loss_plot("test_loss_curve.png")
    print(f"  Saved: {path}")

    print("Generating gradient norm plot...")
    path = logger.generate_gradient_norm_plot("test_gradient_norms.png")
    print(f"  Saved: {path}")

    print("Generating summary JSON...")
    path = logger.export_summary_json("test_summary.json")
    print(f"  Saved: {path}")

    print("Generating all diagnostics...")
    paths = logger.generate_all_diagnostics()
    print(f"  Generated {len(paths)} diagnostic files")

    return True


def test_telemetry_with_model():
    print("\n" + "=" * 60)
    print("TEST: Telemetry with Real Model")
    print("=" * 60)

    import torch
    from pipeline.training.telemetry import PrecisionTelemetryLogger
    from pipeline.models.architectures import AAVTropismTransformer

    output_dir = os.path.join(os.path.dirname(__file__), "..", "diagnostics", "test_model")
    logger = PrecisionTelemetryLogger(output_dir)

    model = AAVTropismTransformer()

    print("Generating weight distribution plot...")
    path = logger.generate_weight_distribution(model, "test_weights.png")
    print(f"  Saved: {path}")

    for epoch in range(5):
        logger.log_weight_stats(epoch, model)

    print("Generating weight evolution plot...")
    path = logger.generate_weight_evolution_plot("test_weight_evolution.png")
    print(f"  Saved: {path}")

    print("Generating all diagnostics with model...")
    paths = logger.generate_all_diagnostics(model)
    print(f"  Generated {len(paths)} diagnostic files")

    return True


def test_bayesian_convergence():
    print("\n" + "=" * 60)
    print("TEST: Bayesian Convergence Plotting")
    print("=" * 60)

    from pipeline.feedback.refinement import LNPRefinery

    output_dir = os.path.join(os.path.dirname(__file__), "..", "diagnostics", "test_bayesian")
    refinery = LNPRefinery(diagnostics_dir=output_dir)

    import numpy as np

    def dummy_objective(params):
        return np.random.uniform(0.3, 0.9)

    print("Running small Bayesian optimization...")
    result = refinery.optimize(dummy_objective)

    print(f"Best score: {result.best_score:.4f}")
    print(f"Iterations: {result.iteration_count}")

    print("Saving convergence plot...")
    path = refinery.save_convergence_plot("test_convergence.png")
    print(f"  Saved: {path}")

    print("Saving parameter importance plot...")
    path = refinery.save_param_importance_plot("test_param_importance.png")
    print(f"  Saved: {path}")

    print("Saving exploration vs exploitation plot...")
    path = refinery.save_exploration_vs_exploitation("test_exploration.png")
    print(f"  Saved: {path}")

    return True


if __name__ == "__main__":
    print("TELEMETRY INTEGRATION TESTS")
    print("=" * 60)

    results = []
    results.append(("Telemetry Logger", test_telemetry_logger()))
    results.append(("Telemetry with Model", test_telemetry_with_model()))
    results.append(("Bayesian Convergence", test_bayesian_convergence()))

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
        print("ALL TELEMETRY TESTS PASSED!")
        print("\nDiagnostic files saved to:")
        print("  - diagnostics/test/")
        print("  - diagnostics/test_model/")
        print("  - diagnostics/test_bayesian/")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
