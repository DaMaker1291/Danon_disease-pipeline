import os
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logger.warning("scikit-optimize not available. Using custom Bayesian optimization.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Convergence plots disabled.")

try:
    from pipeline.models.osk_safety import OSKDoxycyclineSwitch, OSKPenalizedFitness
    HAS_OSK_SAFETY = True
except ImportError:
    HAS_OSK_SAFETY = False
    logger.warning("OSK safety module not available. Skipping OSK constraints.")


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    iteration_count: int
    convergence_history: list[float]
    param_history: list[dict]
    noise_estimate: float = 0.0


class BayesianRefinery:
    def __init__(self, param_space: dict = None, n_initial: int = 20,
                 n_iterations: int = 100, noise_alpha: float = 0.1,
                 diagnostics_dir: str = "./diagnostics"):
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.noise_alpha = noise_alpha
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(diagnostics_dir, exist_ok=True)
        self.convergence_history = []
        self.param_history = []
        self.score_history = []

        self.param_space = param_space or {
            "ionizable_frac": {"type": "float", "low": 0.30, "high": 0.50},
            "peg_frac": {"type": "float", "low": 0.005, "high": 0.025},
            "cholesterol_frac": {"type": "float", "low": 0.25, "high": 0.40},
            "pka": {"type": "float", "low": 5.8, "high": 6.8},
            "tail_length": {"type": "int", "low": 10, "high": 22},
            "unsaturation": {"type": "int", "low": 0, "high": 5},
        }

    def optimize(self, objective_fn, initial_points=None, initial_scores=None) -> OptimizationResult:
        if HAS_SKOPT:
            return self._optimize_with_skopt(objective_fn, initial_points, initial_scores)
        return self._optimize_custom(objective_fn, initial_points, initial_scores)

    def _optimize_with_skopt(self, objective_fn, initial_points, initial_scores) -> OptimizationResult:
        dimensions = []
        param_names = []
        for name, spec in self.param_space.items():
            param_names.append(name)
            if spec["type"] == "float":
                dimensions.append(Real(spec["low"], spec["high"], name=name))
            elif spec["type"] == "int":
                dimensions.append(Integer(spec["low"], spec["high"], name=name))

        @use_named_args(dimensions)
        def objective(**params):
            params["tail_length"] = int(params.get("tail_length", 16))
            params["unsaturation"] = int(params.get("unsaturation", 2))
            score = objective_fn(params)
            self.score_history.append(score)
            self.param_history.append(params.copy())
            return -score

        x0 = None
        if initial_points:
            x0 = [[p[name] for name in param_names] for p in initial_points[:self.n_initial]]

        result = gp_minimize(
            objective, dimensions, n_calls=self.n_iterations,
            x0=x0, random_state=42, n_initial_points=self.n_initial,
        )

        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        best_params["tail_length"] = int(best_params.get("tail_length", 16))
        best_params["unsaturation"] = int(best_params.get("unsaturation", 2))

        return OptimizationResult(
            best_params=best_params, best_score=-result.fun,
            iteration_count=len(result.func_vals),
            convergence_history=[-v for v in result.func_vals],
            param_history=self.param_history,
            noise_estimate=self.noise_alpha,
        )

    def _optimize_custom(self, objective_fn, initial_points, initial_scores) -> OptimizationResult:
        rng = np.random.RandomState(42)

        if initial_points and initial_scores:
            X = np.array([[p[name] for name in self.param_space.keys()] for p in initial_points])
            y = np.array(initial_scores)
        else:
            X = self._random_sample(rng, self.n_initial)
            y = np.array([objective_fn(self._to_dict(x)) for x in X])

        for iteration in range(self.n_iterations - len(y)):
            next_x = self._acquisition(X, y, rng)
            next_score = objective_fn(self._to_dict(next_x))
            X = np.vstack([X, next_x.reshape(1, -1)])
            y = np.append(y, next_score)
            self.score_history.append(next_score)
            self.param_history.append(self._to_dict(next_x))

        best_idx = np.argmax(y)
        best_x = X[best_idx]
        best_params = self._to_dict(best_x)

        if len(y) > 1:
            noise_est = np.std(y) * self.noise_alpha
        else:
            noise_est = self.noise_alpha

        return OptimizationResult(
            best_params=best_params, best_score=y[best_idx],
            iteration_count=len(y), convergence_history=y.tolist(),
            param_history=self.param_history, noise_estimate=noise_est,
        )

    def _random_sample(self, rng, n) -> np.ndarray:
        samples = []
        for name, spec in self.param_space.items():
            if spec["type"] == "float":
                samples.append(rng.uniform(spec["low"], spec["high"], size=n))
            else:
                samples.append(rng.randint(spec["low"], spec["high"] + 1, size=n))
        return np.column_stack(samples)

    def _to_dict(self, x: np.ndarray) -> dict:
        params = {}
        for i, (name, spec) in enumerate(self.param_space.items()):
            if spec["type"] == "int":
                params[name] = int(x[i])
            else:
                params[name] = float(x[i])
        return params

    def _acquisition(self, X, y, rng) -> np.ndarray:
        best_y = np.max(y)
        mu, sigma = self._gaussian_process(X, y, X[-1:])

        improvement = mu - best_y
        Z = improvement / max(sigma, 1e-8)
        ei = improvement * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)

        candidate_x = self._random_sample(rng, 1000)
        candidate_mu, candidate_sigma = self._gaussian_process(X, y, candidate_x)

        candidate_improvement = candidate_mu - best_y
        candidate_Z = candidate_improvement / np.maximum(candidate_sigma, 1e-8)
        candidate_ei = (candidate_improvement * self._norm_cdf(candidate_Z) +
                        candidate_sigma * self._norm_pdf(candidate_Z))

        best_idx = np.argmax(candidate_ei)
        return candidate_x[best_idx]

    def _gaussian_process(self, X_train, y_train, X_test):
        theta = 1.0
        length_scale = 1.0

        sq_dist = np.sum((X_train[:, None] - X_train[None, :]) ** 2, axis=2)
        K_train = theta * np.exp(-0.5 * sq_dist / length_scale ** 2)
        K_train += (self.noise_alpha ** 2 + 1e-6) * np.eye(len(K_train))

        sq_dist_test = np.sum((X_test[:, None] - X_train[None, :]) ** 2, axis=2)
        K_test = theta * np.exp(-0.5 * sq_dist_test / length_scale ** 2)

        K_ss = theta * np.ones(len(X_test)) + self.noise_alpha ** 2

        try:
            K_inv = np.linalg.inv(K_train)
            mu = K_test @ K_inv @ y_train
            sigma = K_ss - np.sum(K_test @ K_inv * K_test, axis=1)
            sigma = np.maximum(sigma, 1e-8)
        except np.linalg.LinAlgError:
            mu = np.zeros(len(X_test))
            sigma = np.ones(len(X_test))

        return mu, np.sqrt(sigma)

    def _norm_cdf(self, x):
        return 0.5 * (1 + np.vectorize(lambda t: np.math.erf(t / np.sqrt(2)))(x))

    def _norm_pdf(self, x):
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def save_convergence_plot(self, filename="bayesian_convergence.png"):
        if not self.score_history or not HAS_MATPLOTLIB:
            return None

        fig, ax1 = plt.subplots(figsize=(10, 5))

        best_so_far = []
        current_best = float("inf")
        for s in self.score_history:
            current_best = min(current_best, s)
            best_so_far.append(current_best)

        ax1.plot(self.score_history, alpha=0.3, color="steelblue", label="Per-Iteration Score")
        ax1.plot(best_so_far, color="crimson", lw=2, label="Best Score So Far")

        if self.noise_alpha > 0:
            noise_bands = [self.noise_alpha * 0.5] * len(self.score_history)
            upper = [b + n for b, n in zip(best_so_far, noise_bands)]
            lower = [b - n for b, n in zip(best_so_far, noise_bands)]
            ax1.fill_between(range(len(best_so_far)), lower, upper,
                           alpha=0.1, color="crimson", label="Noise Band (α)")

        ax1.axvline(x=self.n_initial, color="forestgreen", linestyle="--",
                    alpha=0.7, label="Initial Random → Bayesian")

        ax1.set_title("Bayesian Optimization Convergence (OSK-Penalized Safety)",
                      fontsize=11, fontweight="bold")
        ax1.set_xlabel("Function Evaluation Step", fontsize=10)
        ax1.set_ylabel("Objective Score", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        path = os.path.join(self.diagnostics_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[Telemetry] Saved Bayesian convergence plot: %s", path)
        return path

    def save_param_importance_plot(self, filename="param_importance.png"):
        if not self.param_history or not self.score_history or not HAS_MATPLOTLIB:
            return None

        param_names = list(self.param_space.keys())
        n_params = len(param_names)
        if n_params == 0:
            return None

        fig, axes = plt.subplots(1, min(n_params, 4), figsize=(14, 4))
        if n_params == 1:
            axes = [axes]

        X = np.array([[p.get(k, 0) for k in param_names] for p in self.param_history])
        y = np.array(self.score_history)

        for idx, pname in enumerate(param_names[:4]):
            ax = axes[idx]
            param_vals = X[:, idx]
            scatter = ax.scatter(param_vals, y, c=y, cmap="RdYlGn_r", s=20, alpha=0.6)
            ax.set_title(pname, fontsize=9, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Score" if idx == 0 else "", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)

        for idx in range(min(n_params, 4), len(axes)):
            axes[idx].axis("off")

        fig.suptitle("Parameter vs. Fitness Scatter", fontsize=11, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.diagnostics_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[Telemetry] Saved parameter importance plot: %s", path)
        return path

    def save_exploration_vs_exploitation(self, filename="exploration_exploitation.png"):
        if len(self.score_history) < self.n_initial + 2 or not HAS_MATPLOTLIB:
            return None

        exploration = self.score_history[:self.n_initial]
        exploitation = self.score_history[self.n_initial:]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(range(len(exploration)), exploration, color="dodgerblue",
               alpha=0.7, label="Exploration (Random Init)")
        offset = len(exploration)
        ax.bar(range(offset, offset + len(exploitation)), exploitation,
               color="crimson", alpha=0.7, label="Exploitation (Bayesian GP)")

        ax.axvline(x=offset - 0.5, color="black", linestyle="--", lw=2)
        ax.set_title("Exploration vs. Exploitation Phase", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Step", fontsize=10)
        ax.set_ylabel("Objective Score", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend()

        path = os.path.join(self.diagnostics_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[Telemetry] Saved exploration vs exploitation plot: %s", path)
        return path


class AAVRefinery(BayesianRefinery):
    def __init__(self, noise_alpha: float = 0.15, diagnostics_dir: str = "./diagnostics"):
        super().__init__(
            param_space={
                "mutation_positions": {"type": "int", "low": 263, "high": 732},
                "aa_substitution": {"type": "int", "low": 0, "high": 19},
                "mutation_count": {"type": "int", "low": 1, "high": 15},
            },
            n_initial=30, n_iterations=200, noise_alpha=noise_alpha,
            diagnostics_dir=diagnostics_dir,
        )


class LNPRefinery(BayesianRefinery):
    def __init__(self, noise_alpha: float = 0.10, diagnostics_dir: str = "./diagnostics"):
        super().__init__(
            param_space={
                "ionizable_frac": {"type": "float", "low": 0.30, "high": 0.50},
                "peg_frac": {"type": "float", "low": 0.005, "high": 0.025},
                "cholesterol_frac": {"type": "float", "low": 0.25, "high": 0.40},
                "pka": {"type": "float", "low": 5.8, "high": 6.8},
                "tail_length": {"type": "int", "low": 10, "high": 22},
                "unsaturation": {"type": "int", "low": 0, "high": 5},
            },
            n_initial=30, n_iterations=200, noise_alpha=noise_alpha,
            diagnostics_dir=diagnostics_dir,
        )


class FeedbackLoop:
    def __init__(self, aav_refinery: AAVRefinery = None, lnp_refinery: LNPRefinery = None,
                 osk_max_dox_days: float = 56, osk_penalty_weight: float = 0.3):
        self.aav_refinery = aav_refinery or AAVRefinery()
        self.lnp_refinery = lnp_refinery or LNPRefinery()
        self.osk_max_dox_days = osk_max_dox_days
        self.osk_penalty_weight = osk_penalty_weight
        if HAS_OSK_SAFETY:
            self.osk_penalizer = OSKPenalizedFitness(max_days=osk_max_dox_days,
                                                       penalty_weight=osk_penalty_weight)
        else:
            self.osk_penalizer = None

    def refine_from_sequencing(self, winners: list, aav_generator, lnp_generator, filter_fn) -> dict:
        logger.info("Starting feedback refinement from %d winners", len(winners))

        aav_winners = [w for w in winners if hasattr(w, "sequence")]
        lnp_winners = [w for w in winners if hasattr(w, "ionizable_lipid")]

        refined_aav = None
        if aav_winners:
            refined_aav = self._refine_aav(aav_winners, aav_generator, filter_fn)

        refined_lnp = None
        if lnp_winners:
            refined_lnp = self._refine_lnp(lnp_winners, lnp_generator, filter_fn)

        return {"aav_refinement": refined_aav, "lnp_refinement": refined_lnp}

    def _refine_aav(self, winners, generator, filter_fn):
        def objective(params):
            candidates = generator.generate_candidates(0, 100)
            for c in candidates:
                pos = params["mutation_positions"]
                if pos < len(c.sequence):
                    aa_idx = params["aa_substitution"]
                    c.sequence = c.sequence[:pos] + "ACDEFGHIKLMNPQRSTVWY"[aa_idx] + c.sequence[pos+1:]
                c.osk_expression_level = params.get("osk_expression_level", 0.5)
                c.doxycycline_days = params.get("doxycycline_days", 42)
            scored = generator.score_candidates(candidates)
            filtered = filter_fn(scored)
            if not filtered:
                return 0.0
            fitnesses = []
            for c in filtered:
                base_fitness = c.fitness
                if self.osk_penalizer:
                    penalized = self.osk_penalizer(
                        base_fitness,
                        getattr(c, "osk_expression_level", 0.5),
                        getattr(c, "doxycycline_days", 42),
                    )
                    fitnesses.append(penalized)
                else:
                    fitnesses.append(base_fitness)
            return np.mean(fitnesses)

        initial = []
        initial_scores = []
        for w in winners[:20]:
            if hasattr(w, "mutations") and w.mutations:
                for pos, orig, new in w.mutations[:3]:
                    initial.append({
                        "mutation_positions": pos,
                        "aa_substitution": "ACDEFGHIKLMNPQRSTVWY".index(new) if new in "ACDEFGHIKLMNPQRSTVWY" else 0,
                        "mutation_count": 1,
                    })
                    initial_scores.append(getattr(w, "fitness", 0.5))

        return self.aav_refinery.optimize(objective, initial, initial_scores)

    def _refine_lnp(self, winners, generator, filter_fn):
        def objective(params):
            c = generator._generate_single(np.random.RandomState(0), 0)
            c.ionizable_frac = params["ionizable_frac"]
            c.peg_frac = params["peg_frac"]
            c.cholesterol_frac = params["cholesterol_frac"]
            c.pka = params["pka"]
            c.tail_length = int(params["tail_length"])
            c.unsaturation = int(params["unsaturation"])
            c.helper_frac = 1.0 - c.ionizable_frac - c.peg_frac - c.cholesterol_frac
            c.osk_expression_level = params.get("osk_expression_level", 0.5)
            c.doxycycline_days = params.get("doxycycline_days", 42)
            scored = generator.score_candidates([c])
            if not scored:
                return 0.0
            base_fitness = scored[0].fitness
            if self.osk_penalizer:
                return self.osk_penalizer(
                    base_fitness,
                    getattr(scored[0], "osk_expression_level", 0.5),
                    getattr(scored[0], "doxycycline_days", 42),
                )
            return base_fitness

        initial = []
        initial_scores = []
        for w in winners[:20]:
            if hasattr(w, "composition_dict"):
                comp = w.composition_dict()
                comp["osk_expression_level"] = getattr(w, "osk_expression_level", 0.5)
                comp["doxycycline_days"] = getattr(w, "doxycycline_days", 42)
                initial.append(comp)
                initial_scores.append(getattr(w, "fitness", 0.5))

        return self.lnp_refinery.optimize(objective, initial, initial_scores)
