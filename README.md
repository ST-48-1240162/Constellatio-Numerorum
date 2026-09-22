# Constellatio Numerorum

![algebraic numbers, full view](algebraics/algebraics_h17_3840px_default.png)

![algebraic numbers, detail near (0.86, 0.58)](algebraics/algebraics_h17_wiki_crop.png)

![random stable IIR poles, orders 2 to 18](iir-poles/poles_o18_3840px.png)

Stephen J. Brooks's [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number) sketch, reimplemented as additive Lorentzian blobs in the complex plane.

```
common/              splat + batched float64 roots
algebraics/          integer-polynomial roots, constellatio annotation
algebraics/regions/  single-colour full-map exports (deg1 to deg9_plus)
iir-poles/           random Schur-stable IIR poles
```

`requirements.txt`: numpy, pygame, Pillow, matplotlib. GPU: PyTorch + CUDA. PDF outputs also need `pdflatex` (TeX Live).

## Colour

Both plots share one hue table: the same `o` always gives the same base RGB from `DEGREE_COLORS` in `common/render.py` (`color_for_degree`). What `o` counts depends on the plot. Brightness and mark size do not.

For `color_for_degree`, indices `o = 1` to `8` map to `DEGREE_COLORS[o]`. For `o ≥ 9` (and any unlisted slot) the colour is **white**. Each point stores `o` in `points["o"]` or `pts["o"]`.

<table>
  <thead>
    <tr>
      <th>colour</th>
      <th><code>o</code></th>
      <th>algebraic plots</th>
      <th>IIR plot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span style="background-color:#FF0000;color:#fff;padding:2px 8px;">red</span> 1.00, 0.00, 0.00</td>
      <td>1</td>
      <td>degree-1 integer polynomial</td>
      <td>order 1</td>
    </tr>
    <tr>
      <td><span style="background-color:#00FF00;color:#000;padding:2px 8px;">green</span> 0.00, 1.00, 0.00</td>
      <td>2</td>
      <td>degree 2</td>
      <td>order 2</td>
    </tr>
    <tr>
      <td><span style="background-color:#0000FF;color:#fff;padding:2px 8px;">blue</span> 0.00, 0.00, 1.00</td>
      <td>3</td>
      <td>degree 3</td>
      <td>order 3</td>
    </tr>
    <tr>
      <td><span style="background-color:#B3B300;color:#000;padding:2px 8px;">olive</span> 0.70, 0.70, 0.00</td>
      <td>4</td>
      <td>degree 4</td>
      <td>order 4</td>
    </tr>
    <tr>
      <td><span class="hue hue-orange">orange</span> 1.00, 0.60, 0.00</td>
      <td>5</td>
      <td>degree 5</td>
      <td>order 5</td>
    </tr>
    <tr>
      <td><span class="hue hue-cyan">cyan</span> 0.00, 1.00, 1.00</td>
      <td>6</td>
      <td>degree 6</td>
      <td>order 6</td>
    </tr>
    <tr>
      <td><span class="hue hue-magenta">magenta</span> 1.00, 0.00, 1.00</td>
      <td>7</td>
      <td>degree 7</td>
      <td>order 7</td>
    </tr>
    <tr>
      <td><span class="hue hue-grey">grey</span> 0.60, 0.60, 0.60</td>
      <td>8</td>
      <td>degree 8</td>
      <td>order 8</td>
    </tr>
    <tr>
      <td><span class="hue hue-white">white</span> 1.00, 1.00, 1.00</td>
      <td>≥9</td>
      <td>degree 9 and above</td>
      <td>order 9+ (if generated)</td>
    </tr>
  </tbody>
</table>

Algebraic plots: $`o = k = \deg p`$ for the enumerated $`p(z)=\sum_{n=0}^{k} c_n z^n`$ with $`c_k>0`$. The colour tags the polynomial you enumerated. A cubic in the list is blue at every root, even roots that also satisfy a linear or quadratic.

IIR plot: $`o = M = \deg A`$ for $`A(z)=1+\sum_{m=1}^{M} a_m z^{-m}`$ from reflection coeffs $`k_0,\ldots,k_{M-1}`$.

Brightness and mark width use different formulas on each plot:

| plot | brightness | mark width |
| --- | --- | --- |
| algebraic blobs | hit count $`w`$ at the same rounded $`(x,y,h,o)`$, splat amplitude $`w \cdot C(o)`$ | blob radius $`r = k_1 k_2^{h-3}`$, lower Brooks $`h`$, wider glow |
| IIR poles | $`0.55 \cdot \mathrm{clip}(\log(1+Q)/6.5,\ 0.06,\ 1) \cdot C(o)`$, higher $`Q`$ when $`|z|`$ is closer to 1 | $`k_1 k_2^{M-2}`$, wider at higher order |

Brooks complexity (algebraic plots):

$$
h = \sum_{n=0}^{k} \left(|c_n| + 1\right)
$$

IIR pole sharpness:

$$
Q \approx \frac{0.5\,|z|}{1-|z|}
$$

## Algebraic numbers

Defaults $k_1=0.125$, $k_2=0.5$. Leading coeff $c_k>0$, degree $k\ge 1$. Each $h$ is a unary bit encoding of $|c_n|$, then all sign patterns on the non-leading nonzero coeffs. CPU: `numpy.roots`. GPU: batched float64 companion eigenvalues. Hits at the same $(x,y,h)$ and degree merge into one blob with $`w>1`$.

Viewport is three numbers passed to `render`: centre $`(ox, oy)`$ in the complex plane and `zoom` (pixels per unit). Default centres on the origin with `zoom = height / 5`.

```bash
# full view (top image)
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --height 2160 \
  --png algebraics/algebraics_h17_3840px_default.png

# detail crop like [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png) (second image)
python3 algebraics/algebraics_gpu.py --maxh 17 --width 1920 --height 1080 \
  --ox 0.86 --oy 0.58 --zoom 820 \
  --png algebraics/algebraics_h17_wiki_crop.png
```

For another width, scale `zoom` with width. At 3840 px wide the same crop uses `--zoom 1640` ($820 \times 3840/1920$).

CPU path:

```bash
python3 algebraics/algebraics.py --maxh 15 --png algebraics/algebraics.png
```

Colab: `algebraics/colab_algebraics.ipynb`. Brooks used `maxh=15`. T4: 17, A100: 18.

`algebraics/algebraics_h17_3840px_default.png` (T4, batch 8192, 3840×2160, default viewport): 803744 polys, 5744032 roots, 4660457 unique blobs, GPU roots 492 s.

## Constellatio study

![constellatio annotated map, h=17, 7680×4320](algebraics/constellatio_annotated.png)

Two related outputs share the same root cache (`--maxh` must match everywhere):

1. **Annotated map**: one labelled full render (`constellatio.pdf`).
2. **Single-colour plots**: nine full renders, one hue-table entry each (`deg1` to `deg9_plus`).

Callout geometry and labels live in `algebraics/regions/defs.py` (`CALLOUT_REGIONS`). Degree-plot IDs live in the same file (`DEGREE_PLOTS`).

### Annotated map

```bash
python3 algebraics/algebraics_gpu.py --maxh 17 --width 7680 --height 4320 \
  --png algebraics/algebraics_h17_7680px.png

python3 algebraics/annotate_constellatio.py algebraics/algebraics_h17_7680px.png
```

### Single-colour degree plots

```bash
python3 algebraics/regions/build.py
# subset:  python3 algebraics/regions/build.py --plots deg3 deg6
# skip PDF: python3 algebraics/regions/build.py --no-pdf
```

Reads the consolidated blob cache (no PNG input). Default **7680×4320**, viewport `centre (0,0)`, `zoom = height/5`. Writes one folder per plot:

```
algebraics/regions/deg1/     deg1.png  deg1.pdf  points.npz  summary.txt
algebraics/regions/deg2/
...
algebraics/regions/deg9_plus/
```

<table>
  <thead>
    <tr>
      <th>plot</th>
      <th>degree</th>
      <th>colour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>deg1</code></td>
      <td>1</td>
      <td><span style="background-color:#FF0000;color:#fff;padding:2px 8px;">red</span></td>
    </tr>
    <tr>
      <td><code>deg2</code></td>
      <td>2</td>
      <td><span style="background-color:#00FF00;color:#000;padding:2px 8px;">green</span></td>
    </tr>
    <tr>
      <td><code>deg3</code></td>
      <td>3</td>
      <td><span style="background-color:#0000FF;color:#fff;padding:2px 8px;">blue</span></td>
    </tr>
    <tr>
      <td><code>deg4</code></td>
      <td>4</td>
      <td><span style="background-color:#B3B300;color:#000;padding:2px 8px;">olive</span></td>
    </tr>
    <tr>
      <td><code>deg5</code></td>
      <td>5</td>
      <td><span class="hue hue-orange">orange</span></td>
    </tr>
    <tr>
      <td><code>deg6</code></td>
      <td>6</td>
      <td><span class="hue hue-cyan">cyan</span></td>
    </tr>
    <tr>
      <td><code>deg7</code></td>
      <td>7</td>
      <td><span class="hue hue-magenta">magenta</span></td>
    </tr>
    <tr>
      <td><code>deg8</code></td>
      <td>8</td>
      <td><span class="hue hue-grey">grey</span></td>
    </tr>
    <tr>
      <td><code>deg9_plus</code></td>
      <td>≥9</td>
      <td><span class="hue hue-white">white</span></td>
    </tr>
  </tbody>
</table>

Any plot can still show clipped white/gold where many blobs of **that same degree** overlap.

Colab one-shot: `algebraics/colab_algebraics.ipynb`, or `algebraics/run_colab_cli.sh` (GPU render + degree plots + constellatio PDF → downloads `constellatio.pdf` and `constellatio_regions.zip`).

## Random stable IIR poles

Reflection coeffs with $`|k|<1`$. Levinson step-up to Schur polynomials. High order piles up near $`|z|=1`$.

```bash
python3 iir-poles/lattice_poles.py --orders 2-16 --n 350
python3 iir-poles/lattice_poles.py --gpu --orders 2-18 --n 1500 --width 3840 --height 2160 --png iir-poles/poles_o18_3840px.png
```

`iir-poles/poles_o18_3840px.png` (T4, orders 2-18): ~197k poles, median $`|z|`$ 0.76 (order 2) to 0.999 (order 18), ~1 min GPU. Colab: `iir-poles/colab_poles.ipynb`.
