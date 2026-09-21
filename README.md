# Constellatio Numerorum

![algebraic numbers at maxh=17](algebraics/algebraics_h17_3840px.png)

![algebraic plane scatter, coeffs in [-4,4]](algebraics/algebraic_plane_readme.png)

![random stable IIR poles, orders 2 to 18](iir-poles/poles_o18_3840px.png)

Stephen J. Brooks's [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number) sketch, reimplemented as additive Lorentzian blobs in the complex plane.

```
common/        splat + batched float64 roots
algebraics/    integer-polynomial roots
iir-poles/     random Schur-stable IIR poles
```

`requirements.txt`: numpy, pygame, Pillow, matplotlib. GPU: PyTorch + CUDA.

## How it works

```mermaid
flowchart LR
  subgraph inputs [inputs]
    E1[Brooks bit enum]
    E2[coeff box enum]
    E3[random reflection k]
  end

  subgraph roots [root finding]
    R1[numpy.roots CPU]
    R2[gpu_roots batched eig]
  end

  subgraph render [render]
    V1[Lorentzian splat]
    V2[matplotlib scatter]
  end

  E1 --> R1
  E1 --> R2
  E2 --> R1
  E2 --> R2
  E3 --> R1
  E3 --> R2
  R1 --> V1
  R2 --> V1
  R1 --> V2
  R2 --> V2
  V1 --> PNG[PNG]
  V2 --> PNG
```

Three entry scripts share `common/` (`gpu_roots.py`, `render.py`). They enumerate or sample polynomials, find roots in the complex plane, then draw with Brooks colours.

## Colour

`color_for_degree` in `common/render.py` picks RGB from `DEGREE_COLORS[o]`. Each point stores `o` in `points["o"]` or `pts["o"]`.

| colour | `o` | algebraic plots | IIR plot |
| --- | --- | --- | --- |
| red | 1 | degree-1 integer polynomial | order-1 filter |
| green | 2 | degree 2 | order 2 |
| blue | 3 | degree 3 | order 3 |
| olive | 4 | degree 4 | order 4 |
| orange | 5 | degree 5 | order 5 |
| cyan | 6 | degree 6 | order 6 |
| magenta | 7 | degree 7 | order 7 |
| grey | 8 | degree 8 | order 8 |
| white | 0, ≥9 | slot unused; degree/order 9+ | same |

Algebraic plots: $`o = k = \deg p`$ for the enumerated $`p(z)=\sum_{n=0}^{k} c_n z^n`$ with $`c_k>0`$. The colour tags the polynomial you enumerated. A cubic in the list is blue at every root, even roots that also satisfy a linear or quadratic.

IIR plot: $`o = M = \deg A`$ for $`A(z)=1+\sum_{m=1}^{M} a_m z^{-m}`$ from reflection coeffs $`k_0,\ldots,k_{M-1}`$.

Brightness and mark width use different formulas on each plot:

| plot | brightness | mark width |
| --- | --- | --- |
| algebraic blobs | hit count $`w`$ at the same rounded $`(x,y,h,o)`$; splat amplitude $`w \cdot C(o)`$ | blob radius $`r = k_1 k_2^{h-3}`$; lower Brooks $`h`$, wider glow |
| algebraic scatter | alpha $`\propto k_2^{h-3}`$ and $`\log(1+w)`$ (default `--weight both`) | dot area $`\propto (\max|c_i|)^{-1}`$ and $`\sqrt{w}`$ |
| IIR poles | $`0.55 \cdot \mathrm{clip}(\log(1+Q)/6.5,\ 0.06,\ 1) \cdot C(o)`$; higher $`Q`$ when $`|z|`$ is closer to 1 | $`k_1 k_2^{M-2}`$; wider at higher order |

Brooks complexity (algebraic plots):

$$
h = \sum_{n=0}^{k} \left(|c_n| + 1\right)
$$

IIR pole sharpness:

$$
Q \approx \frac{0.5\,|z|}{1-|z|}
$$

## Algebraic numbers

```mermaid
flowchart TD
  A["--maxh N"] --> B["enumerate_abs_coeffs\nBrooks unary-bit patterns"]
  B --> C["signed_coeffs / signed_matrix\nall sign patterns, ck > 0"]
  C --> D{CPU or GPU?}
  D -->|algebraics.py| E["numpy.roots per poly"]
  D -->|algebraics_gpu.py| F["roots_batched\nfloat64 companion eig"]
  E --> G["points x,y,h,o"]
  F --> G
  G --> H{cache points_hN.npz?}
  H -->|hit| I[load]
  H -->|miss| J["consolidate\nmerge x,y,h,o → weight w"]
  J --> K[save npz]
  I --> L[render]
  K --> L
  L --> M["C = DEGREE_COLORS o\namplitude = w"]
  L --> N["radius r = k1·k2^ h-3"]
  M --> O["splat additive Lorentzian"]
  N --> O
  O --> P["PNG or pygame viewer"]
```

Defaults $k_1=0.125$, $k_2=0.5$. Leading coeff $c_k>0$, degree $k\ge 1$. Each $h$ is a unary bit encoding of $|c_n|$, then all sign patterns on the non-leading nonzero coeffs. CPU: `numpy.roots`. GPU: batched float64 companion eigenvalues. Hits at the same $(x,y,h)$ and degree merge into one blob with $`w>1`$.

```bash
python3 algebraics/algebraics.py --maxh 15
python3 algebraics/algebraics.py --maxh 15 --view wiki --png algebraics/algebraics.png
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --png algebraics/algebraics_h17_3840px.png
```

`--view wiki`: `ox=0.86`, `oy=0.58`, `zoom=820` (1920×1080), matching [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png). Colab: `algebraics/colab_algebraics.ipynb`. Brooks used `maxh=15`; T4: 17, A100: 18.

`algebraics/algebraics_h17_3840px.png` (T4, batch 8192, 3840×2160): 803744 polys, 5744032 roots, 4660457 unique blobs, GPU roots 499 s.

## Algebraic plane (scatter)

```mermaid
flowchart TD
  A[algebraic_plane.py] --> B{--enum}
  B -->|box| C["deg 1..D, coeffs in -L..L\nmeshgrid all polys"]
  B -->|brooks| D["collect_by_degree up to maxh\nneeds --gpu"]
  C --> E{CPU or GPU?}
  D --> E
  E -->|CPU| F["numpy.roots"]
  E -->|GPU| G["roots_batched"]
  F --> H["filter to xlim/ylim"]
  G --> H
  H --> I["consolidate_roots\nx,y,deg → count w"]
  I --> J{cache plane enum tag.npz?}
  J -->|hit| K[load]
  J -->|miss| L[save]
  K --> M[render_scatter]
  L --> M
  M --> N["scatter_metrics\nsize + alpha from --weight"]
  N --> O["matplotlib scatter\nlow degree on top"]
  O --> P[PNG]
```

Same hue table as above. Low degree drawn on top.

- `--weight height`: size $`\propto 1/\max|c_i|`$, alpha $`\propto 1/\max|c_i|`$
- `--weight h`: size and alpha both $`\propto k_2^{h-3}`$
- `--weight both` (default): size from coefficient height, alpha from Brooks $`h`$; hit count $`w`$ boosts both

`--enum box`: integer polys with coeffs in $`[-L,L]`$. `--enum brooks`: Brooks bit encoding, needs `--gpu`. Cache: `algebraics/cache/plane_{enum}_{tag}.npz`; `--recompute` to rebuild.

```bash
python3 algebraics/algebraic_plane.py --png algebraics/algebraic_plane.png
python3 algebraics/algebraic_plane.py --gpu --coeff-range 4 --max-degree 5 --png algebraics/algebraic_plane_3840px.png
python3 algebraics/algebraic_plane.py --gpu --enum brooks --maxh 12 --weight h --png algebraics/algebraic_plane_brooks.png
```

Second image: 1920 resize of `algebraic_plane_3840px.png` (Colab T4, `coeff-range=4`, 2.53M roots, ~4.6 min). Default `coeff-range=3`: 222k blobs.

## Random stable IIR poles

```mermaid
flowchart TD
  A["--orders M1-M2, --n filters/order"] --> B["for each order M"]
  B --> C["draw K ~ U(-0.995,0.995)^M"]
  C --> D["step_up_batch / reflection_to_direct\nLevinson → A(z)"]
  D --> E{CPU or GPU?}
  E -->|CPU| F["numpy.roots"]
  E -->|GPU| G["roots_batched"]
  F --> H["keep poles inside unit disk"]
  G --> H
  H --> I["pole_q from abs z\npts x,y,o=M,q"]
  I --> J["draw unit circle ring"]
  I --> K["C = DEGREE_COLORS M"]
  I --> L["brightness 0.55·clip log1+Q /6.5"]
  I --> M["radius k1·k2^ M-2"]
  J --> N[splat]
  K --> N
  L --> N
  M --> N
  N --> O[PNG]
```

Reflection coeffs with $`|k|<1`$; Levinson step-up to Schur polynomials. High order piles up near $`|z|=1`$.

```bash
python3 iir-poles/lattice_poles.py --orders 2-16 --n 350
python3 iir-poles/lattice_poles.py --gpu --orders 2-18 --n 1500 --width 3840 --height 2160 --png iir-poles/poles_o18_3840px.png
```

`iir-poles/poles_o18_3840px.png` (T4, orders 2-18): ~197k poles, median $`|z|`$ 0.76 (order 2) to 0.999 (order 18), ~1 min GPU. Colab: `iir-poles/colab_poles.ipynb`.
