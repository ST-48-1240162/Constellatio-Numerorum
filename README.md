# Constellatio Numerorum

![algebraic numbers at maxh=17](algebraics/algebraics_h17_3840px.png)

![algebraic plane scatter, coeffs in [-4,4]](algebraics/algebraic_plane_readme.png)

![random stable IIR poles, orders 2 to 18](iir-poles/poles_o18_3840px.png)

This repo follows Stephen J. Brooks's original code on the Wikipedia page for [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number). The plots are additive Lorentzian blobs in the complex plane. One folder per plot.

```
common/        splat + batched float64 roots
algebraics/    integer-polynomial roots
iir-poles/     random Schur-stable IIR poles
```

`requirements.txt` has numpy, pygame, Pillow, matplotlib. GPU needs PyTorch + CUDA. Run from the repo root.

Colour is the integer field `o` (`points["o"]` / `pts["o"]`), looked up by `color_for_degree` in `common/render.py`:

- algebraic: $`o = k = \deg p`$ for the enumerated $`p(z)=\sum_{n=0}^{k} c_n z^n`$ ($`c_k>0`$, $`k\ge 1`$), not $`\min\{\deg q:q(z)=0\}`$
- IIR: $`o = M`$ for $`A(z)=1+\sum_{m=1}^{M} a_m z^{-m}`$, from reflection coeffs $`k_0,\ldots,k_{M-1}`$
- $`C(o)=\texttt{DEGREE\_COLORS}[o]`$ if $`1\le o\le 8`$, else white $`(1,1,1)`$
- red $`o=1`$, `1.00, 0.00, 0.00`
- green $`o=2`$, `0.00, 1.00, 0.00`
- blue $`o=3`$, `0.00, 0.00, 1.00`
- olive $`o=4`$, `0.70, 0.70, 0.00`
- orange $`o=5`$, `1.00, 0.60, 0.00`
- cyan $`o=6`$, `0.00, 1.00, 1.00`
- magenta $`o=7`$, `1.00, 0.00, 1.00`
- grey $`o=8`$, `0.60, 0.60, 0.60`
- white $`o=0`$ (unused slot) and $`o\ge 9`$

## Algebraic numbers

- $`C = C(o)`$, $`o=k=\deg p`$ (red $`k=1`$, green $`k=2`$, blue $`k=3`$, …)
- RGB is multiplied by hit count times blob size, so overlaps get brighter

Blob radius falls with Brooks complexity $h$:

$$
h = \sum_{n=0}^{k} \left(|c_n| + 1\right), \qquad r = k_1 k_2^{h-3}
$$

Defaults $k_1=0.125$, $k_2=0.5$. Leading coeff $c_k>0$, degree $k\ge 1$. Each $h$ is a unary bit encoding of $|c_n|$, then all sign patterns on the non-leading nonzero coeffs. CPU roots use `numpy.roots`; GPU roots are batched float64 companion eigenvalues. Overlapping hits at the same $(x,y,h)$ and degree merge.

```bash
python3 algebraics/algebraics.py --maxh 15
python3 algebraics/algebraics.py --maxh 15 --view wiki --png algebraics/algebraics.png
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --png algebraics/algebraics_h17_3840px.png
```

`--view wiki` is `ox=0.86`, `oy=0.58`, `zoom=820` (1920×1080), matching [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png). Colab notebook: `algebraics/colab_algebraics.ipynb`. Brooks used `maxh=15`. T4 gets to 17, A100 to 18.

`algebraics/algebraics_h17_3840px.png` (T4, batch 8192, 3840×2160): 803744 polys, 5744032 roots, 4660457 unique blobs, GPU roots 499 s.

## Algebraic plane (scatter)

Matplotlib scatter of algebraic roots in the complex plane.

- same $`C(o)`$ with $`o=k=\deg p`$
- default `--weight both`: dot size from coefficient height $`\max|c_i|`$, alpha from Brooks $`h`$ and $`k_2`$
- roots at the same rounded $`(x,y)`$ and degree merge; hit count boosts size and alpha
- low degree is drawn last

`--enum box` enumerates all integer polys with coeffs in $`[-L,L]`$ (the default). `--enum brooks` uses Brooks bit encoding and needs `--gpu`. Roots cache to `algebraics/cache/plane_{enum}_{tag}.npz` unless you pass `--recompute`.

```bash
python3 algebraics/algebraic_plane.py --png algebraics/algebraic_plane.png
python3 algebraics/algebraic_plane.py --gpu --coeff-range 4 --max-degree 5 --png algebraics/algebraic_plane_3840px.png
python3 algebraics/algebraic_plane.py --gpu --enum brooks --maxh 12 --weight h --png algebraics/algebraic_plane_brooks.png
```

The second image is a 1920 resize of `algebraic_plane_3840px.png` (Colab T4, `coeff-range=4`, 2.53M roots, ~4.6 min). Local default (`coeff-range=3`) is 222k blobs.

## Random stable IIR poles

Draw reflection coeffs with $`|k|<1`$. Levinson step-up gives Schur polynomials, so the poles sit inside the unit circle. High order piles up near $`|z|=1`$.

- $`C = C(o)`$, $`o=M=\deg A`$ (green $`M=2`$, blue $`M=3`$, …; white $`M\ge 9`$)
- that colour is then scaled by $`0.55 \cdot \mathrm{clip}(\log(1+Q)/6.5,\ 0.06,\ 1)`$
- $`Q \approx 0.5\,r/(1-r)`$, $`r=|z|`$

```bash
python3 iir-poles/lattice_poles.py --orders 2-16 --n 350
python3 iir-poles/lattice_poles.py --gpu --orders 2-18 --n 1500 --width 3840 --height 2160 --png iir-poles/poles_o18_3840px.png
```

Colab notebook: `iir-poles/colab_poles.ipynb`.

`iir-poles/poles_o18_3840px.png` (T4, orders 2-18): ~197k poles, median $`|z|`$ 0.76 (order 2) to 0.999 (order 18), ~1 min GPU.
