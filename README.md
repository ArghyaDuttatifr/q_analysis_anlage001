Analyzing temperature-dependent resonance sweeps requires extracting $Q_L$, $Q_{ex}$, and $Q_{in}$ while isolating line attenuation and crosstalk. Doing this sequentially on single threads causes severe performance bottlenecks when handling large datasets.

## Paper for the Fitting Model
Measurement of resonant frequency and quality factor of microwave resonators: Comparison of methods by Paul J. Petersan; Steven M. Anlage


## 🔬 Mathematical Fitting Model

The fitting algorithm models the linear transmission magnitude $|S_{21}(f)|$ of a microwave resonator using a **real Lorentzian resonance model** superimposed on a **linear background**. To prevent floating-point underflow during non-linear optimization, all frequencies are internally scaled to gigahertz (GHz), and the peak amplitude ($S_m$) is fixed prior to fitting.

### 1. Transmission Model Equation
The linear transmission response as a function of frequency $f$ (in GHz) is modeled as:

$$|S_{21}(f)| = \frac{S_m + A_3 f}{1 + 4 \left(\frac{f - f_0}{\Delta f}\right)^2} + (A_1 + A_2 f)$$

### 2. Parameter Definitions
| Parameter | Symbol | Description |
| :--- | :--- | :--- |
| **Resonant Frequency** | $f_0$ ($\omega_r$) | Center frequency of energy storage peak (GHz). |
| **Linewidth (FWHM)** | $\Delta f$ (`del_f`) | The full width at half maximum of the resonance peak (GHz). |
| **Peak Amplitude** | $S_m$ | The isolated amplitude of the resonance, fixed automatically as max ($S_{12}$). |
| **Linear Background** | $A_1, A_2$ | The constant baseline offset ($A_1$) and linear slope ($A_2$) of the feedline. |
| **Peak Skew** | $A_3$ | Linear skew term for frequency-dependent amplitude scaling. |

### 3. Extraction of Quality Factors
Unlike complex models that fit $Q_L$ directly, this application fits the physical linewidth ($\Delta f$) and derives the quality factors sequentially:

1. **Loaded $Q$-Factor ($Q_L$):** Represents total system losses.
   $$Q_L = \frac{f_0}{\Delta f}$$

2. **External $Q$-Factor ($Q_{ex}$):** Represents coupling losses to the feedlines.
   $$Q_{ex} = \frac{Q_L}{S_m}$$

3. **Internal $Q$-Factor ($Q_{in}$):** The intrinsic losses of the device, extracted by removing external coupling:
   $$Q_{in} = \left( \frac{1}{Q_L} - \frac{1}{Q_{ex}} \right)^{-1}$$

### 4. Optimization Strategy
To avoid local minima in steep resonance landscapes, a two-stage strategy is used:
1. **Grid Search over $f_0$:** The center frequency $f_0$ is evaluated over $N_{\text{trials}}$ fixed points within a narrow window around the raw peak frequency.
2. **Sub-Optimization:** For each trial $f_0$, non-linear least squares (`scipy.optimize.curve_fit`) optimizes the 4 free parameters $[A_1, A_2, A_3, \Delta f]$ within strict parameter bounds.
