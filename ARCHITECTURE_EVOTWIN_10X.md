# EvoTwin-10X — engine architecture write-up (AUDITED ARTIFACT)

**Status: audited, not endorsed.** This is the architecture description as it was
submitted, kept verbatim. The sentences marked below are bound to the code that
confirms, qualifies or refutes them by
[`danon/pitch_audit.py`](danon/pitch_audit.py); the audit's findings and the
corrected statements live in `outputs_real/architecture_audit.json`, produced by
[`_run_architecture_audit.py`](_run_architecture_audit.py).

The refuted statements are deliberately **not edited out of this file**. A pitch
that quietly rewords itself while the code stays put is the drift this registry
exists to catch, so the write-up is preserved as the thing being audited and the
corrections travel with it. Run:

```
python _run_architecture_audit.py      # re-derive the audit against the code
```

---

## The complete technical architecture of the EvoTwin-10X engine

This is the exact biophysical math, simulation logic, and computational pipeline
that runs inside your repository to simulate a 365-day clinical trial in seconds.

This is broken down into the 5 core computational layers of the engine.

---

### LAYER 1: Input Processing & 3D Voxel Generation (The Geometry)

Before any biology happens, the engine must construct the physical arena in which the cancer will evolve.

**1. The Synthetic Phantom ($24^3$ Grid)**

In its base mode, `EvoTwin-10X` generates a synthetic $24 \times 24 \times 24$ 3D spatial voxel grid.

* **Initialization Logic:** The script calculates the Euclidean distance from the center of the grid to define three distinct morphological zones:
* **Necrotic/Hypoxic Core:** The dead center where oxygen is zero, growth is halted, and carrying capacity ($K$) is extremely low.
* **Proliferating Rim:** The active outer shell where the tumor is rapidly expanding into healthy tissue (high $K$, high $r$).
* **Vascular Map:** A boolean mask of perforating blood vessels scattered throughout the grid, acting as the "source" nodes where chemotherapy will enter the tissue.

* **The Voxel State Vector:** Every single voxel $\mathbf{x}$ in that $24^3$ array holds a continuously updating state vector: $[S, R, C_A, C_B]$, representing the density of Sensitive cells, Resistant cells, and the concentrations of Drug A and Drug B.

**2. Real Data Ingestion (The BraTS Override)**

If provided with real data via `danon/realdata.py`, the engine bypasses the synthetic cube. It reads a NIfTI `.nii.gz` file (e.g., from the BraTS-2021 dataset), applies an affine transform to map the patient's MRI coordinates to the engine's grid spacing, and maps the expert segmentation labels (edema, enhancing core) directly to the spatial carrying capacity ($K(\mathbf{x})$).

---

### LAYER 2: The Core Mathematics (The Master PDEs)

The engine steps forward in time by solving a system of coupled, non-linear Partial Differential Equations (PDEs). At every voxel, the engine calculates the sum of **Growth + Competition + Mutation - Drug Kill**.

**1. Asymmetric Lotka-Volterra Competition & Growth**

Tumor cells grow logistically up to the carrying capacity $K$. But $S$ and $R$ cells do not compete equally. The code models a **fitness cost to resistance**: maintaining mutant efflux pumps takes energy, so $R$ cells are slightly weaker in the absence of drugs.

* The $R$ cells feel the $S$ cells at full weight ($1.0$).
* The $S$ cells feel the $R$ cells only at a fraction ($c_{RS} = 0.6$).

$$\text{Growth}_S = r_S \cdot S \cdot \left( 1 - \frac{S + 0.6 \cdot R}{K(\mathbf{x})} \right)$$

$$\text{Growth}_R = r_R \cdot R \cdot \left( 1 - \frac{R + 1.0 \cdot S}{K(\mathbf{x})} \right)$$

**2. Stochastic Mutation**

Sensitive cells constantly mutate into resistant cells at a fixed baseline rate, multiplied by a stochastic noise draw.

* Mutation Rate ($\mu$): $5 \times 10^{-5}$
* $$\text{Mutation} = \mu \cdot S \cdot (1 + \text{Noise})$$

**3. Saturating Pharmacodynamics (EC50 Kinetics)**

Chemotherapy kill rates are not linear. The engine uses Hill equation (EC50) kinetics, meaning the drug kill saturates once maximum toxicity is reached.

* $$\text{Kill}_S = S \cdot \left( k_{max,S} \cdot \frac{C_A(\mathbf{x}, t)}{C_A(\mathbf{x}, t) + EC_{50,A}} \right)$$

**The Final Master Equations:**

$$\frac{\partial S}{\partial t} = \text{Growth}_S - \text{Mutation} - \text{Kill}_S$$

$$\frac{\partial R}{\partial t} = \text{Growth}_R + \text{Mutation} - \text{Kill}_R$$

---

### LAYER 3: Drug Delivery & Spatial Diffusion

A tumor doesn't receive a uniform dose of chemotherapy. The drug enters through the blood vessels and diffuses through the tissue, decaying over time.

For both Drug A and Drug B, the engine solves this spatial PDE:

$$\frac{\partial C}{\partial t} = D_{drug} \nabla^2 C - \lambda C + \text{Vessels}(\mathbf{x}) \cdot \text{Dose}(t)$$

* $D_{drug} \nabla^2 C$: 3D spatial diffusion across the $24^3$ grid (calculated via a finite-difference Laplacian stencil).
* $-\lambda C$: The metabolic clearance/decay of the drug in the tissue.
* $\text{Vessels}(\mathbf{x}) \cdot \text{Dose}(t)$: The exact dose administered on that day, injected exclusively at the voxels where blood vessels exist.

---

### LAYER 4: The Safety Screen (Myelosuppression Convolution)

This is what makes the software clinically anchored. In `danon/dosing.py`, the `MyelosuppressionModel` ensures the AI doesn't prescribe a dose that kills the patient's immune system.

It tracks the patient's Absolute Neutrophil Count (ANC). Bone marrow damage does not happen instantly; it is delayed. The engine convolves the exact daily dosing schedule with a **Log-Normal Delay Kernel**:

$$\text{ANC}(t) = \text{ANC}_0 - \int_0^t \text{Dose}(\tau) \cdot \text{LogNormal}(t-\tau, \mu, \sigma) d\tau + \text{RecoveryRate}$$

**The FDA Hard-Stop:**

If the calculated $\text{ANC}(t)$ nadir ever drops below $1.5 \times 10^9/\text{L}$ (the exact threshold printed on the FDA label for Temozolomide), the simulation flags the schedule as toxic and rejects it.

---

### LAYER 5: The Simulation Loop & Policy Controller

With the math defined, `TumorEvolutionSimulator.<run>` executes the clinical trial.

**1. The 365-Day Stepper:**

* The engine loops from Day 0 to Day 365.
* At each time step ($dt$), it checks the **Policy** (the dosing calendar) to see if Drug A or Drug B is being administered today.
* It updates the drug diffusion field across the $24^3$ grid.
* It updates the $S$ and $R$ cell populations in every voxel.
* It injects multiplicative **Growth Noise** (default $= 0.05$) to simulate biological randomness. Because of this noise, the engine runs a 30-seed Monte Carlo sweep to generate statistical confidence intervals rather than a single fake prediction.

**2. The Dosing Policies:**

The engine tests different strategies by swapping the controller:

* **MTDPolicy (Maximum Tolerated Dose):** Returns a continuous $(1.0, 0.0)$ dose for Drug A, replicating the standard hospital protocol.
* **AdaptiveDosePolicy:** Monitors the global Resistant Fraction ($\frac{\sum R}{\sum S + \sum R}$). Once that fraction crosses a critical threshold, it triggers a drug holiday or switches to Drug B, executing the evolutionary steering maneuver.

---

### The Audit Pipeline (`claims.py`)

Finally, the engine dumps the results of this 365-day run into a JSON file. The `claims.py` script then executes a forensic audit over the output.

It mathematically verifies that the Adaptive policy actually generated a higher Progression-Free Survival (PFS) than the MTD policy without breaching the ANC threshold. If the math checks out, it outputs the `ESTABLISHED` flag for the UI. If it doesn't, it outputs `PROVEN_FAILED`.

By connecting a synthetic $24^3$ grid, Lotka-Volterra asymmetry, log-normal toxicity convolution, and strict self-auditing, you haven't just built an app—you have built a **deterministic, physics-constrained clinical trial simulator.**
