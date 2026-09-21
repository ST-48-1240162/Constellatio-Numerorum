# Constellatio Numerorum

![algebraic numbers at maxh=17](algebraics/algebraics_h17_3840px.png)

![random stable IIR poles, orders 2 to 18](iir-poles/poles_o18_3840px.png)

This repo is following Stephen J. Brooks's original code posted on the Wikipedia page of [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number). Additive Lorentzian blobs in the complex plane. One folder per plot.

```
common/        splat + batched float64 roots
algebraics/    integer-polynomial roots
iir-poles/     random Schur-stable IIR poles
```

`requirements.txt`: numpy, pygame, Pillow. GPU: PyTorch + CUDA. Run from repo root.

## Algebraic numbers

Color is degree. Blob radius falls with Brooks complexity $`h`$:

```math
h = \sum_{n=0}^{k} \bigl(|c_n| + 1\bigr), \quad r = k_1 k_2^{h-3}
```

Defaults $`k_1=0.125`$, $`k_2=0.5`$. Leading coeff $`c_k>0`$, degree $`k\ge 1`$. Each $`h`$ is a unary bit encoding of $`|c_n|`$, then all sign patterns on non-leading nonzero coeffs. Roots: `numpy.roots` (CPU) or batched float64 companion eigenvalues (GPU). Overlapping hits at the same $`(x,y,h)`$ and degree merge. Brightness is hit count times blob size.

```bash
python3 algebraics/algebraics.py --maxh 15
python3 algebraics/algebraics.py --maxh 15 --view wiki --png algebraics/algebraics.png
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --png algebraics/algebraics_h17_3840px.png
```

`--view wiki`: `ox=0.86`, `oy=0.58`, `zoom=820` (1920×1080), matching [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png). Colab: `algebraics/colab_algebraics.ipynb`. Brooks used `maxh=15`. T4: 17. A100: 18.

**Sample** (`algebraics/algebraics_h17_3840px.png`, T4, batch 8192, 3840×2160): 803744 polys, 5744032 roots, 4660457 unique blobs, GPU roots 499 s.

## Random stable IIR poles

Draw reflection coeffs with $`|k|<1`$. Levinson step-up gives Schur polynomials, poles inside the unit circle. Color is order. Brightness is pole $`Q \approx 0.5\,r/(1-r)`$ ($`r=|z|`$). High order piles up near $`|z|=1`$.

```bash
python3 iir-poles/lattice_poles.py --orders 2-16 --n 350
python3 iir-poles/lattice_poles.py --gpu --orders 2-18 --n 1500 --width 3840 --height 2160 --png iir-poles/poles_o18_3840px.png
```

Colab: `iir-poles/colab_poles.ipynb`.

**Sample** (`iir-poles/poles_o18_3840px.png`, T4, orders 2-18): ~197k poles, median $`|z|`$ 0.76 (order 2) to 0.999 (order 18), ~1 min GPU.
