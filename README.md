# Non-Uniform Fast Fourier Transform for Two-Electron Repulsion Integrals

Two-electron repulsion integrals (ERIs) constitute a fundamental computational bottleneck in quantum chemistry, scaling as $O(N^4)$ with the number of basis functions. This Projects presents implementation and analysis of ERI computation using the Non-Uniform Fast Fourier Transform (NUFFT) on adaptive Becke grids. We develop a modular six-component framework encompassing Gaussian-type orbital evaluation, numerical grid generation, orbital product computation, Fourier transformation, two-electron integral calculation, and systematic method comparison. Numerical validation demonstrates that all basis functions (s, p, d types) satisfy normalization to within $10^{-15}$ to $10^{-10}$ relative error, overlap integrals achieve $3.16 \times 10^{-10}$ accuracy against analytical values, and NUFFT-based Fourier transforms agree with analytical results to $10^{-12}$. For ERI computation, NUFFT on Becke grids achieves a relative error of $4.54 \times 10^{-6}$ using 7,760 grid points, while conventional FFT on uniform grids yields $1.05 \times 10^{-1}$ relative error (10.5%) with 884,736 points. This corresponds to a 23,038-fold improvement in accuracy using 114 times fewer grid points. Direct real-space integration fails with errors exceeding $10^{6}$ due to the Coulomb singularity at $r_{12} = 0$, which Fourier methods handle analytically through the $4\pi/k^2$ kernel. All results are validated against analytical reference values computed via the Boys function.

## Headline Result

| Method | Grid Points | Relative Error | Significant Digits |
|--------|:-----------:|:--------------:|:------------------:|
| **NUFFT (Becke)** | **7,760** | $4.54 \times 10^{-6}$ | **~6** |
| FFT (Uniform) | 884,736 | $1.05 \times 10^{-1}$ | ~1 |
| Direct | 3,300 | $1.15 \times 10^{7}$ | **FAILED** |

**NUFFT achieves 23,038× better accuracy with 114× fewer grid points** compared to FFT.

## The Problem

Two-electron repulsion integrals are the computational bottleneck of quantum chemistry, scaling as $O(N^4)$:

$$(\mu\nu|\sigma\lambda) = \int\!\!\int \frac{\chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)\chi_\sigma(\mathbf{r}_2)\chi_\lambda(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} \, d\mathbf{r}_1 \, d\mathbf{r}_2$$

The Coulomb operator $1/|\mathbf{r}_1 - \mathbf{r}_2|$ has a singularity at $r_{12}=0$ that destroys direct numerical integration. The key insight is the Fourier representation:

$$\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} = \frac{1}{2\pi^2} \int \frac{e^{i\mathbf{k}\cdot(\mathbf{r}_1 - \mathbf{r}_2)}}{k^2} \, d\mathbf{k}$$

which converts the 6D real-space integral into a 3D Fourier-space integral with the well-behaved $4\pi/k^2$ kernel:

$$(\mu\nu|\sigma\lambda) = \frac{1}{2\pi^2} \int \frac{\tilde{\rho}_{\mu\nu}(\mathbf{k}) \cdot \tilde{\rho}^*_{\sigma\lambda}(\mathbf{k})}{k^2} \, d\mathbf{k}$$

## Dependencies

| Package | Purpose |
|---------|---------|
| NumPy / SciPy | Array operations, FFT, `erf` |
| `grid-qchem` | Gauss-Legendre, Becke transforms, AtomGrid, MolGrid, BeckeWeights |
| `FINUFFT` | Type-3 NUFFT via `finufft.nufft3d3()` |

---

##  GTO Evaluation

### What It Does

Implements Cartesian Gaussian-Type Orbitals for s-type ($l=0$), p-type ($l=1$), and d-type ($l=2$) angular momenta, including contracted basis functions (STO-3G).

A Cartesian GTO centered at $\mathbf{R}$ is:

$$\chi(\mathbf{r}) = N \cdot (x - R_x)^{i_x} (y - R_y)^{i_y} (z - R_z)^{i_z} \cdot e^{-\alpha|\mathbf{r}-\mathbf{R}|^2}$$

with normalization constant:

$$N = \left(\frac{2\alpha}{\pi}\right)^{3/4} \cdot (4\alpha)^{l/2} \cdot \frac{1}{\sqrt{(2i_x-1)!! \cdot (2i_y-1)!! \cdot (2i_z-1)!!}}$$

### Implementation overview

**Primitive GTO evaluation** — vectorized over all grid points simultaneously:
1. Compute displacement vectors $\mathbf{d} = \text{points} - \text{center}$
2. Compute $r^2 = d_x^2 + d_y^2 + d_z^2$
3. Radial part: `exp(-α * r²)`
4. Angular part: $d_x^{i_x} \cdot d_y^{i_y} \cdot d_z^{i_z}$
5. Return $N \cdot \text{angular} \cdot \text{radial}$

**Contracted GTO (STO-3G)** — sums $K$ primitives: $\chi = \sum_{k=1}^K c_k N_k e^{-\alpha_k r^2}$

**Shell evaluation** — allocates output array of shape `(ncomp, npts)` where $n_\text{comp} = (l+1)(l+2)/2$, fills each row with the contracted GTO for the corresponding $(i_x, i_y, i_z)$ tuple (enumerated by looping $i_x$ from $l$ down to 0, $i_y$ from $l - i_x$ down to 0, $i_z = l - i_x - i_y$).

**Double factorial** — iterative: start at $k = 2n-1$, multiply into accumulator, decrement by 2 until $k \leq 0$.

### Test Results 

**Test 1.0 — Double Factorial Verification**

| $n$ | Computed $(2n-1)!!$ | Expected |
|:---:|:---:|:---:|
| 0 | 1 | 1 |
| 1 | 1 | 1 |
| 2 | 3 | 3 |
| 3 | 15 | 15 |
| 4 | 105 | 105 |


**Test 1.1 — GTO Normalization** (Grid: $50^3 = 125{,}000$ pts, $L=20$ Bohr, $h=0.4082$ Bohr)

| Orbital | $(i_x,i_y,i_z)$ | $N$ | $\int\|\chi\|^2 d\mathbf{r}$ | Error |
|:---:|:---:|:---:|:---:|:---:|
| s | (0,0,0) | 0.712705 | 1.000000 | $8.21\times10^{-13}$ |
| s ($\alpha\!=\!0.5$) | (0,0,0) | 0.423777 | 1.000000 | $2.11\times10^{-15}$ |
| $p_x$ | (1,0,0) | 1.425411 | 1.000000 | $1.54\times10^{-11}$ |
| $p_y$ | (0,1,0) | 1.425411 | 1.000000 | $1.54\times10^{-11}$ |
| $p_z$ | (0,0,1) | 1.425411 | 1.000000 | $1.54\times10^{-11}$ |
| $d_{x^2}$ | (2,0,0) | 1.645923 | 1.000000 | $2.88\times10^{-10}$ |
| $d_{xy}$ | (1,1,0) | 2.850822 | 1.000000 | $3.16\times10^{-11}$ |
| $d_{xz}$ | (1,0,1) | 2.850822 | 1.000000 | $3.16\times10^{-11}$ |
| $d_{y^2}$ | (0,2,0) | 1.645923 | 1.000000 | $2.88\times10^{-10}$ |
| $d_{yz}$ | (0,1,1) | 2.850822 | 1.000000 | $3.16\times10^{-11}$ |
| $d_{z^2}$ | (0,0,2) | 1.645923 | 1.000000 | $2.88\times10^{-10}$ |



**Test 1.2 — Gaussian Decay** ($\alpha=1.0$, $N=0.712705$)

| $r$ (Bohr) | $\chi(r)$ | Region |
|:---:|:---:|:---|
| 0.0 | $7.127\times10^{-1}$ | Maximum at nucleus |
| 1.0 | $2.516\times10^{-1}$ | Core |
| 2.0 | $1.107\times10^{-2}$ | Valence |
| 3.1 | $6.069\times10^{-5}$ | Outer valence |
| 4.1 | $4.147\times10^{-8}$ | Asymptotic tail |

Confirms $e^{-\alpha r^2}$ behavior; negligible beyond 4 Bohr. 

**Test 1.3 — Angular Nodal Structure**

| Test | Result | Status |
|------|--------|:---:|
| $p_x$ at $x=0$ plane | $\chi = [0, 0, 0, 0]$ | 
| $p_x$ sign change: $(1,0,0)$ vs $(-1,0,0)$ | $[+0.524, -0.524]$ | 
| $d_{xy}$ quadrant symmetry | $[+0.386, -0.386, +0.386, -0.386]$ | 

**Test 1.4 — Contracted GTO (STO-3G Hydrogen)**

| Primitive | $\alpha_k$ | $c_k$ |
|:---:|:---:|:---:|
| 1 | 3.42525091 | 0.15432897 |
| 2 | 0.62391373 | 0.53532814 |
| 3 | 0.16885540 | 0.44463454 |

$\int\|\chi_\text{STO-3G}\|^2 d\mathbf{r} = 0.999984$ (finite grid error). Maximum at origin. 

**Test 1.5 — d-Shell Orthogonality**

| | $d_{z^2}$ | $d_{yz}$ | $d_{y^2}$ | $d_{xz}$ | $d_{xy}$ | $d_{x^2}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $d_{z^2}$ | 1.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.333 |
| $d_{yz}$ | — | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| $d_{y^2}$ | — | — | 1.000 | 0.000 | 0.000 | 0.333 |
| $d_{xz}$ | — | — | — | 1.000 | 0.000 | 0.000 |
| $d_{xy}$ | — | — | — | — | 1.000 | 0.000 |
| $d_{x^2}$ | — | — | — | — | — | 1.000 |

The overlap of $1/3$ between e.g. $d_{x^2}$ and $d_{y^2}$ is expected — Cartesian $x^2 e^{-\alpha r^2}$ and $y^2 e^{-\alpha r^2}$ are **not** the orthogonal spherical harmonic d-orbitals. The value arises from $\int x^2 y^2 e^{-2\alpha r^2} d\mathbf{r} \neq 0$.

---

##  Grid Generation

### What It Does

Constructs two types of quadrature grids: **uniform Cartesian grids** (FFT-compatible) and **adaptive atom-centered Becke grids** (for NUFFT).

### Implementation overview

**Uniform grid** — fills a cubic box $[-L/2, L/2]^3$ with spacing $h = L/N$:
- 1D coordinates via `linspace(-L/2, L/2, N, endpoint=False)` (endpoint excluded for FFT periodicity)
- 3D tensor product via `meshgrid` → $N^3$ total points, all weights = $h^3$
- Constructor variants: `from_box`, `from_center`, `from_molecule` (auto-sizes around atomic coordinates with padding)

**Becke grid** — atom-centered spherical grid:
1. **Radial:** Gauss-Legendre points $\{x_i, w_i^{GL}\}$ on $[-1,1]$, mapped via Becke transform $r_i = R(1+x_i)/(1-x_i)$ with transformed weights $w_i^r = w_i^{GL} \cdot 2R/(1-x_i)^2$
2. **Angular:** Lebedev quadrature on the unit sphere (50–770 points, integrating spherical harmonics up to $l=11$–$29$)
3. **Combined:** $n_r \times n_\Omega$ points per atom: $\mathbf{r}_{ij} = r_i \hat{\Omega}_j + \mathbf{R}_A$, $w_{ij} = w_i^r \cdot w_j^\Omega \cdot r_i^2$
4. **Molecular partitioning (Becke weights):** For each atom pair, compute confocal elliptical coordinate $\mu_{AB} = (|\mathbf{r}-\mathbf{R}_A| - |\mathbf{r}-\mathbf{R}_B|)/|\mathbf{R}_A - \mathbf{R}_B|$, then smooth step function $s(\mu) = \tfrac{1}{2}[1-f(f(f(\mu)))]$ where $f(x) = (3x-x^3)/2$. The triple application of $f$ ensures smooth partition of unity.

### Test Results 

**Test 2.1 — Uniform Grid Properties** ($L=10$ Bohr, $N=21$)

| Property | Value |
|----------|:---:|
| Total points | 9,261 |
| Origin | $(-5,-5,-5)$ Bohr |
| Spacing $h$ | 0.5 Bohr |
| Weight per point | $h^3 = 0.125$ Bohr³ |

**Test 2.2 — Uniform Grid Integration** ($81^3 = 531{,}441$ pts)

| Integral | Numerical | Analytical | Relative Error |
|----------|:---:|:---:|:---:|
| $\int e^{-r^2} d^3\mathbf{r}$ | 5.568328 | 5.568328 | $1.60\times10^{-16}$ |
| $\int (\alpha/\pi)^{3/2} e^{-\alpha r^2} d^3\mathbf{r}$ | 1.000000 | 1.0 | $4.44\times10^{-16}$ |

Machine precision. 

**Test 2.3 — Becke Grid Properties** ($n_\text{rad}=50$, $n_\text{ang}=194$)

| Property | Value |
|----------|:---:|
| Total points | 9,700 |
| Min weight | $3.54\times10^{-11}$ |
| Max weight | $3.41\times10^{9}$ |
| Weight range | $\sim10^{20}$ |
| Min $r$ | 0.0009 Bohr |
| Max $r$ | 2644.9 Bohr |

The 20-order-of-magnitude weight range reflects the $r^2 dr$ volume element: tiny near the nucleus, huge far away (but integrand is zero there).

**Test 2.4 — Becke Integration Accuracy** (target: $\int (\alpha/\pi)^{3/2} e^{-\alpha r^2} d^3\mathbf{r} = 1$)

| $\alpha$ | Grid Points | Result | Absolute Error |
|:---:|:---:|:---:|:---:|
| 0.1 | 30,200 | 1.0000000000 | $2.96\times10^{-14}$ |
| 0.5 | 30,200 | 1.0000000000 | $3.01\times10^{-14}$ |
| 1.0 | 30,200 | 1.0000000000 | $2.96\times10^{-14}$ |
| 2.0 | 30,200 | 1.0000000000 | $2.96\times10^{-14}$ |
| 5.0 | 30,200 | 1.0000000000 | $2.89\times10^{-14}$ |

$10^{-14}$ accuracy regardless of exponent. 

**Test 2.5 — H₂ Molecule Grid**

| Test | Numerical | Expected | Error |
|------|:---:|:---:|:---:|
| $\int[\chi_A^2 + \chi_B^2]d\mathbf{r}$ | 2.00000000 | 2.0 | $1.89\times10^{-10}$ |



**Test 2.6 — Grid Efficiency Comparison** (target: error $< 10^{-5}$)

Uniform grid:

| Spacing (Bohr) | Points | Error | Status |
|:---:|:---:|:---:|:---:|
| 1.00 | 4,913 | $3.10\times10^{-4}$ | 
| 0.50 | 35,937 | $2.22\times10^{-16}$ | 
| 0.40 | 68,921 | $2.22\times10^{-16}$ | 
| 0.30 | 166,375 | $3.80\times10^{-2}$ | 
| 0.25 | 274,625 | 0.00 | 

Becke grid:

| $n_\text{rad}$ | Points | Error | Status |
|:---:|:---:|:---:|:---:|
| 20 | 3,880 | $6.17\times10^{-6}$ | 
| 30 | 5,820 | $2.19\times10^{-8}$ | 
| 40 | 7,760 | $3.01\times10^{-11}$ | 
| 50 | 9,700 | $1.41\times10^{-13}$ | 
| 60 | 11,640 | $1.67\times10^{-15}$ | 

**Becke ~3,880 pts vs Uniform ~35,937 pts → 9.3× fewer points.** The anomalous 0.30 Bohr result (worse than 0.50) is an aliasing artifact — the spacing systematically misses the Gaussian's features.

---

##  Orbital Products

### What It Does

Evaluates basis function products $\chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})$ on numerical grids for overlap integrals and electron densities.

### Implementation overview

**Single product:** Evaluate both GTOs on all grid points, return element-wise product.

**Batch shell-pair products** — vectorized via NumPy broadcasting:
1. Evaluate shell $i$ → shape $(n_i, N_\text{pts})$
2. Evaluate shell $j$ → shape $(n_j, N_\text{pts})$
3. `products = χi[:, newaxis, :] × χj[newaxis, :, :]` → shape $(n_i, n_j, N_\text{pts})$

All $n_i \times n_j$ products computed simultaneously with no Python loops.

**Overlap integral:** $S_{AB} = \sum_k w_k \cdot \chi_A(\mathbf{r}_k) \cdot \chi_B(\mathbf{r}_k)$ — dot product of weighted function values. Analytical reference via the Gaussian product theorem:

$$S_{AB} = N_A N_B \left(\frac{\pi}{\alpha+\beta}\right)^{3/2} \exp\!\left(-\frac{\alpha\beta}{\alpha+\beta}|\mathbf{A}-\mathbf{B}|^2\right)$$

### Test Results

**Test 3.1 — Self-Overlap**

| Grid Points | Numerical | Expected | Error |
|:---:|:---:|:---:|:---:|
| 22,650 | 1.0000000000 | 1.0 | $3.60\times10^{-14}$ |



**Test 3.2 — H₂ Overlap Integral** ($\mathbf{R}_A=(0,0,-0.7)$, $\mathbf{R}_B=(0,0,+0.7)$ Bohr, $\alpha=1.0$)

| Method | Value | Relative Error |
|--------|:---:|:---:|
| Numerical (45,300 pts) | 0.3753110992 | — |
| Analytical | 0.3753110989 | — |
| Difference | — | $3.16\times10^{-10}$ |

$S=0.375$ means 37.5% orbital overlap — characteristic of a covalent bond. 

**Test 3.3 — s-p Orthogonality**

| Integral | Value | Status |
|----------|:---:|:---:|
| $\int \chi_s \cdot \chi_{p_x} d\mathbf{r}$ | $-4.88\times10^{-19}$ | 
| $\int \chi_{p_x}^2 d\mathbf{r}$ | 1.0000000000 | 

Zero by symmetry — $x \cdot e^{-\alpha r^2}$ is odd. 

**Test 3.4 — p-Shell Overlap Matrix**

| | $p_x$ | $p_y$ | $p_z$ |
|:---:|:---:|:---:|:---:|
| $p_x$ | 1.000000 | $3.4\times10^{-18}$ | $4.2\times10^{-18}$ |
| $p_y$ | — | 1.000000 | $2.2\times10^{-18}$ |
| $p_z$ | — | — | 1.000000 |

Max deviation from identity: $3.63\times10^{-14}$. 

**Test 3.5 — STO-3G Hydrogen Self-Overlap**

| Property | Value |
|----------|:---:|
| Grid points | 30,200 |
| $\int \chi^2 d\mathbf{r}$ | 0.9999999909 |

**Test 3.6 — Uniform vs Becke for Density Integration**

Uniform:

| Spacing (Bohr) | Points | Error |
|:---:|:---:|:---:|
| 0.40 | 68,921 | $2.30\times10^{-13}$ |
| 0.30 | 166,375 | $3.80\times10^{-2}$ |
| 0.25 | 274,625 | $1.18\times10^{-14}$ |
| 0.20 | 531,441 | $1.15\times10^{-14}$ |

Becke:

| $n_\text{rad}$ | Points | Error |
|:---:|:---:|:---:|
| 30 | 5,820 | $5.31\times10^{-9}$ |
| 40 | 7,760 | $7.07\times10^{-12}$ |
| 50 | 9,700 | $1.49\times10^{-14}$ |
| 60 | 11,640 | $1.38\times10^{-14}$ |

---

##  Fourier Transforms

### What It Does

Implements three Fourier transform methods: standard FFT on uniform grids, NUFFT Type-3 via FINUFFT on Becke grids, and direct summation for validation.

### Implementation overview

**FFT:** Reshape density to $(N,N,N)$ → 3D FFT → scale by $h^3$ → k-grid from `fftfreq(N, h) × 2π`.

**NUFFT Type-3** — computes $\tilde{f}(\mathbf{k}_j) = \sum_i c_i e^{i\mathbf{k}_j \cdot \mathbf{r}_i}$ where $\{\mathbf{r}_i\}$ are Becke points and $\{\mathbf{k}_j\}$ are spherical k-grid targets:
1. Weight coefficients: $c_i = f(\mathbf{r}_i) \times w_i$
2. Threshold filter: keep only $|c_i| > 10^{-14} \times \max|c|$
3. Cast to `complex128` coefficients, `float64` coordinates (C-contiguous via `ascontiguousarray`)
4. Call `finufft.nufft3d3(x, y, z, c, kx, ky, kz, eps=ε, isign=1)`
5. Fallback to direct summation ($O(MN)$ loop) if FINUFFT fails

FINUFFT achieves $O(N\log N + M)$ via: Gaussian gridding → FFT on oversampled grid → interpolation.

**Spherical k-grid** — Gauss-Legendre in each spherical coordinate, points: $\mathbf{k}_{ijk} = k_i(\sin\phi_k\cos\theta_j, \sin\phi_k\sin\theta_j, \cos\phi_k)$, weights: $w_{ijk} = w_i^r w_j^\theta w_k^\phi k_i^2$.

**Analytical validation transforms:**

$$\tilde{\chi}_s(\mathbf{k}) = N\left(\frac{\pi}{\alpha}\right)^{3/2} e^{-k^2/(4\alpha)} \cdot e^{i\mathbf{k}\cdot\mathbf{R}}, \qquad \tilde{\rho}(\mathbf{k}) = N^2\left(\frac{\pi}{2\alpha}\right)^{3/2} e^{-k^2/(8\alpha)}$$

### Test Results 

**Test 4.1 — Spherical k-Grid** ($10\times10\times10$, $k_\text{max}=10$)

| Property | Value | Expected |
|----------|:---:|:---:|
| Total points | 1,000 | 1,000 |
| $\|k\|$ range | $[0.13, 9.87]$ | $[0, k_\text{max}]$ |
| $\sum w_k$ | 4188.79 | $\frac{4\pi}{3}k_\text{max}^3 = 4188.79$ |
| Relative error | $6.51\times10^{-16}$ | 0 |



**Test 4.2 — NUFFT vs Direct Summation**

| Method | Grid Pts | k-Pts | Time (ms) |
|--------|:---:|:---:|:---:|
| NUFFT | 3,300 | 125 | 25.3 |
| Direct | 3,300 | 125 | 1.06 |

Max difference: $3.49\times10^{-13}$. 

**Test 4.3 — NUFFT on Normalized Gaussian** ($\tilde{f}(\mathbf{k}) = e^{-k^2/(4\alpha)}$)

| Property | Value |
|----------|:---:|
| Grid points | 59,000 |
| k-points | 1,000 |
| Max error | $4.22\times10^{-13}$ |
| Mean error | $9.18\times10^{-14}$ |



**Test 4.4 — NUFFT on s-type GTO**

| Property | Value |
|----------|:---:|
| $\int\|\chi\|^2 d\mathbf{r}$ | 1.0000000000 |
| Grid points | 59,000 |
| Max error vs analytical | $1.67\times10^{-12}$ |
| Mean error | $3.64\times10^{-13}$ |



**Test 4.5 — Off-Center GTO Phase** (center at $(1.0, 0.5, -0.3)$ Bohr)

| | NUFFT | Analytical |
|---|:---:|:---:|
| $\|\tilde{\chi}(\mathbf{k})\|$ at test point | 1.818614 | 1.818614 |
| Max error | $1.49\times10^{-12}$ | — |

Phase factor $e^{i\mathbf{k}\cdot\mathbf{R}}$ correctly captured. 

**Test 4.6 — Fourier Transform of Density** ($\rho = |\chi_s|^2$, effective exponent $2\alpha$)

| Property | Value |
|----------|:---:|
| $\int\|\chi\|^2 d\mathbf{r}$ | 1.0000000000 |
| Max error vs analytical | $4.07\times10^{-13}$ |
| $\tilde{\rho}(k\approx0)$ | 0.998639 |



**Test 4.7 — Convergence**

| $n_\text{rad}$ | $n_\text{ang}$ | Grid Pts | Max Error | Time (ms) |
|:---:|:---:|:---:|:---:|:---:|
| 40 | 194 | 7,760 | $7.14\times10^{-7}$ | 20.9 |
| 60 | 302 | 18,120 | $7.94\times10^{-10}$ | 15.8 |
| 80 | 434 | 34,720 | $1.88\times10^{-12}$ | 25.9 |
| 100 | 590 | 59,000 | $1.88\times10^{-12}$ | 13.3 |
| 120 | 770 | 92,400 | $1.66\times10^{-12}$ | 20.0 |

$10^{-12}$ accuracy at ~35,000 points.

---

##  Two-Electron Integrals

### What It Does

Computes ERIs by combining Fourier-transformed charge densities with the $4\pi/k^2$ Coulomb kernel. Implements FFT, NUFFT, and Direct methods, validated against the analytical Boys function:

$$(ab|cd) = N_a N_b N_c N_d \cdot \frac{2\pi^{5/2}}{\zeta\eta\sqrt{\zeta+\eta}} \cdot K_{ab} K_{cd} \cdot F_0(T), \qquad F_0(T) = \frac{\sqrt{\pi}}{2\sqrt{T}}\text{erf}(\sqrt{T})$$

### Implementation overview

**ERI via NUFFT:**
1. Weight densities: $c = \rho \odot w_\text{grid}$
2. NUFFT Type-3 transform both bra and ket densities to spherical k-grid
3. Coulomb kernel with regularization: $k^2_\text{safe} = \max(k^2, 10^{-20})$
4. Integrate: $\text{ERI} = \text{Re}\!\left(\sum w_k \cdot \tilde{\rho}_\text{bra} \cdot \tilde{\rho}_\text{ket}^* / k^2_\text{safe}\right) / (2\pi^2)$

**ERI via FFT:**
1. Reshape densities to $(N,N,N)$, compute 3D FFTs scaled by $h^3$
2. k=0 singularity: set $k^2[0,0,0]=1$, compute $4\pi/k^2$, then set $\text{Coulomb}[0,0,0]=0$ (charge neutrality)
3. Sum integrand and scale by $\Delta k^3/(2\pi)^3$

**ERI via Direct:** Double loop over grid pairs with $r_{12}^\text{safe} = \max(|\mathbf{r}_1 - \mathbf{r}_2|, 10^{-10})$ — **fails catastrophically** because $1/r_\text{min} = 10^{10}$ corrupts the integral.

### Test Results 

**Test 5.1 — Same-Center (aa|aa)**

| Property | Value |
|----------|:---:|
| Real-space grid | 34,720 points |
| k-space grid | 3,375 points |
| Fourier ERI | 1.1283791672 Ha |
| Analytical | 1.1283791671 Ha |
| Absolute error | $8.69\times10^{-11}$ Ha |
| Relative error | $7.70\times10^{-11}$ |
| Time | 51.8 ms |

**11 significant digits.** 

**Test 5.2 — H₂ Molecule ERIs** ($\mathbf{R}_A=(0,0,-0.7)$, $\mathbf{R}_B=(0,0,+0.7)$; Grid: 69,440 pts)

| Integral | Fourier (Ha) | Analytical (Ha) | Rel. Error |
|:---:|:---:|:---:|:---:|
| $(aa\|aa)$ | 1.12837917 | 1.12837917 | $3.18\times10^{-10}$ |
| $(aa\|bb)$ | 0.68020366 | 0.68020366 | $2.89\times10^{-9}$ |
| $(ab\|ab)$ | 0.15894171 | 0.15894171 | $4.91\times10^{-11}$ |



**Test 5.3 — Fourier vs Direct**

| Method | ERI (Ha) | Error | Time (ms) |
|--------|:---:|:---:|:---:|
| Fourier | 1.12837915 | $1.57\times10^{-8}$ | 32.9 |
| Direct | 5,539,177.89 | $4.91\times10^{6}$ | 546.8 |
| Analytical | 1.12837917 | — | — |

Fourier: 16.6× faster, ~$10^{14}$× more accurate. Direct:  FAILED.

**Test 5.4 — Different Exponents**

| $\alpha_a$ | $\alpha_b$ | $\alpha_c$ | $\alpha_d$ | Fourier (Ha) | Analytical (Ha) | Rel. Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 0.5 | 0.5 | 0.5 | 0.797885 | 0.797885 | $2.82\times10^{-7}$ |
| 1.0 | 1.0 | 1.0 | 1.0 | 1.128379 | 1.128379 | $4.20\times10^{-10}$ |
| 2.0 | 2.0 | 2.0 | 2.0 | 1.595769 | 1.595769 | $1.95\times10^{-9}$ |
| 0.5 | 1.0 | 0.5 | 1.0 | 0.818949 | 0.818949 | $1.25\times10^{-8}$ |
| 1.0 | 2.0 | 1.0 | 2.0 | 1.158169 | 1.158169 | $6.91\times10^{-11}$ |

All ≥ 7 significant digits. 

**Test 5.5 — Convergence (real-space grid, fixed k-grid 3,375 pts)**

| $n_\text{rad}$ | $n_\text{ang}$ | Points | Rel. Error |
|:---:|:---:|:---:|:---:|
| 40 | 194 | 7,760 | $1.68\times10^{-10}$ |
| 60 | 302 | 18,120 | $7.55\times10^{-11}$ |
| 80 | 434 | 34,720 | $7.70\times10^{-11}$ |
| 100 | 590 | 59,000 | $7.41\times10^{-11}$ |

**Convergence (k-space grid, fixed real-space 34,720 pts)**

| $n_k$ | $k_\text{max}$ | k-Points | Rel. Error |
|:---:|:---:|:---:|:---:|
| 8 | 6 | 512 | $2.21\times10^{-5}$ |
| 10 | 8 | 1,000 | $4.77\times10^{-8}$ |
| 12 | 10 | 1,728 | $1.93\times10^{-8}$ |
| 15 | 12 | 3,375 | $4.20\times10^{-10}$ |
| 18 | 15 | 5,832 | $5.81\times10^{-11}$ |

---

##  Method Comparison

### What It Does

Systematic head-to-head benchmarking of all three methods (FFT, NUFFT, Direct) on identical test cases against the analytical Boys function reference.

### Test Results 

**Test 6.1 — Same-Center (ss|ss)** (Analytical: $1.1283791671$ Ha)

FFT (uniform grid):

| $L$ (Bohr) | $N$ | Grid Points | ERI (Ha) | Rel. Error | Time (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 64 | 262,144 | 0.951815 | $1.56\times10^{-1}$ | 20.3 |
| 20 | 80 | 512,000 | 0.986907 | $1.25\times10^{-1}$ | 40.8 |
| 24 | 96 | 884,736 | 1.010386 | $1.05\times10^{-1}$ | 61.4 |
| 28 | 112 | 1,404,928 | 1.027190 | $8.97\times10^{-2}$ | 105.1 |

NUFFT (Becke grid):

| $n_\text{rad}$ | $n_\text{ang}$ | Grid Points | ERI (Ha) | Rel. Error | Time (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 20 | 50 | 1,000 | 1.1283992494 | $1.78\times10^{-5}$ | 21.0 |
| 25 | 86 | 2,150 | 1.1283838002 | $4.11\times10^{-6}$ | 53.5 |
| 30 | 110 | 3,300 | 1.1283842149 | $4.47\times10^{-6}$ | 77.5 |
| 40 | 194 | 7,760 | 1.1283842889 | $4.54\times10^{-6}$ | 163.8 |

Direct:

| Grid Points | ERI (Ha) | Rel. Error | Time (ms) |
|:---:|:---:|:---:|:---:|
| 3,300 | 12,996,684.21 | $1.15\times10^{7}$ | 132.6 |

 **CATASTROPHIC FAILURE** — Coulomb singularity produces error $> 10^7$.

**Test 6.2 — H₂ Molecule** (bond length 1.4 Bohr)

| | FFT (884,736 pts) | | NUFFT (15,520 pts) | |
|:---:|:---:|:---:|:---:|:---:|
| **Integral** | **Value (Ha)** | **Error** | **Value (Ha)** | **Error** |
| $(aa\|aa)$ | 1.01038570 | $1.05\times10^{-1}$ | 1.1283842318 | $4.49\times10^{-6}$ |
| $(aa\|bb)$ | 0.56250863 | $1.73\times10^{-1}$ | 0.6800256196 | $2.62\times10^{-4}$ |
| $(ab\|ab)$ | 0.14232133 | $1.05\times10^{-1}$ | 0.1589424515 | $4.68\times10^{-6}$ |

**Test 6.3 — Direct Comparison**

| Method | Grid Points | ERI (Ha) | Rel. Error | Time (ms) | Status |
|--------|:---:|:---:|:---:|:---:|:---:|
| FFT | 884,736 | 1.01038570 | $1.05\times10^{-1}$ | 64.1 | 
| NUFFT | 7,760 | 1.1283842889 | $4.54\times10^{-6}$ | 159.6 | 
| Direct | 3,300 | 12,996,684.21 | $1.15\times10^{7}$ | 114.8 | 

**Test 6.4 — Convergence**

FFT with box size ($N=96$ fixed):

| $L$ (Bohr) | ERI (Ha) | Rel. Error |
|:---:|:---:|:---:|
| 12 | 0.89375576 | $2.08\times10^{-1}$ |
| 16 | 0.95181507 | $1.56\times10^{-1}$ |
| 20 | 0.98690699 | $1.25\times10^{-1}$ |
| 24 | 1.01038570 | $1.05\times10^{-1}$ |
| 28 | 1.02719023 | $8.97\times10^{-2}$ |
| 32 | 1.03980949 | $7.85\times10^{-2}$ |

FFT error saturates at ~8% due to periodic boundary artifacts and cubic k-grid inefficiency for the spherical $1/k^2$ kernel.

NUFFT with grid refinement:

| $n_\text{rad}$ | $n_\text{ang}$ | Grid Points | Rel. Error |
|:---:|:---:|:---:|:---:|
| 20 | 50 | 1,000 | $1.78\times10^{-5}$ |
| 25 | 86 | 2,150 | $4.11\times10^{-6}$ |
| 30 | 110 | 3,300 | $4.47\times10^{-6}$ |
| 40 | 194 | 7,760 | $4.54\times10^{-6}$ |

---

## Why Each Method Succeeds or Fails

**NUFFT succeeds** — Fourier methods analytically handle the $1/r_{12}$ singularity through the well-behaved $4\pi/k^2$ kernel, while Becke grids concentrate every point where electron density matters.

**FFT fails to converge** — (1) periodic boundary artifacts, (2) finite box truncation, (3) cubic k-grid poorly samples the spherically symmetric $1/k^2$ kernel.

**Direct integration fails catastrophically** — when two grid points coincide, $1/r_\text{min} = 1/10^{-10} = 10^{10}$, which corrupts the entire integral.

## References

1. Szabo A, Ostlund NS. *Modern Quantum Chemistry.* Courier Corporation; 2012.
2. Helgaker T, Jorgensen P, Olsen J. *Molecular Electronic-Structure Theory.* Wiley; 2013.
3. Becke AD. A multicenter numerical integration scheme for polyatomic molecules. *J. Chem. Phys.* 1988;88(4):2547–53.
4. Barnett AH, Magland J, af Klinteberg L. A parallel nonuniform fast Fourier transform library based on an "exponential of semicircle" kernel. *SIAM J. Sci. Comput.* 2019;41(5):C479–504.

## 👤 Author

**Mehdi Hassanbeigi**  
**Email**: hasanbeigimahdi25@gmail.com 




---

##  Copyright Notice

**© 2026 Mehdi. All Rights Reserved.**

**Restrictions**:
- ❌ **No copying, modification, or distribution** of this work is permitted