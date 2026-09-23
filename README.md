# Constellatio Numerorum

![algebraic numbers, full view](algebraics/algebraics_h17_3840px_default.png)

![algebraic numbers, detail near (0.86, 0.58)](algebraics/algebraics_h17_wiki_crop.png)

![random stable IIR poles, orders 2 to 18](iir-poles/poles_o18_3840px.png)

Two labelling methods live on different branches.

Callout map on `master` — yellow leaders and boxes A–E, drawn after the splat by `annotate_constellatio.py`.

![constellatio annotated map, h=17, 7680×4320 (`master`)](algebraics/constellatio_annotated.png)

Inline root labels on `feat/inline-root-labels` — high-hit roots named while the PNG is written (`root_labels.py`).

![algebraic numbers, labelled detail near (0.86, 0.58) (`feat/inline-root-labels`)](algebraics/algebraics_h17_wiki_crop_labeled.png)

This repository reimplements Stephen J. Brooks's [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number) sketch as additive Lorentzian blobs in the complex plane. Brooks' original is [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); this adaptation uses the same license. See [License](#license).

```
common/              splat + batched float64 roots
algebraics/          integer-polynomial roots, constellatio annotation
algebraics/regions/  single-colour full-map exports (deg1 to deg9_plus)
iir-poles/           random Schur-stable IIR poles
```

Install numpy, pygame, Pillow, and matplotlib from `requirements.txt`. GPU rendering needs PyTorch and CUDA. PDF output also needs `pdflatex` from TeX Live.

## Colour

Both plots take hue from `o` through `DEGREE_COLORS` in `common/render.py` (`color_for_degree`). The same `o` always gives the same base RGB. Values from `o = 1` to `8` use the colours below. Values of `o ≥ 9` are white. Each point stores `o` in `points["o"]` or `pts["o"]`.

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
      <td><span style="background-color:#FF9900;color:#000;padding:2px 8px;">orange</span> 1.00, 0.60, 0.00</td>
      <td>5</td>
      <td>degree 5</td>
      <td>order 5</td>
    </tr>
    <tr>
      <td><span style="background-color:#00FFFF;color:#000;padding:2px 8px;">cyan</span> 0.00, 1.00, 1.00</td>
      <td>6</td>
      <td>degree 6</td>
      <td>order 6</td>
    </tr>
    <tr>
      <td><span style="background-color:#FF00FF;color:#fff;padding:2px 8px;">magenta</span> 1.00, 0.00, 1.00</td>
      <td>7</td>
      <td>degree 7</td>
      <td>order 7</td>
    </tr>
    <tr>
      <td><span style="background-color:#999999;color:#fff;padding:2px 8px;">grey</span> 0.60, 0.60, 0.60</td>
      <td>8</td>
      <td>degree 8</td>
      <td>order 8</td>
    </tr>
    <tr>
      <td><span style="background-color:#FFFFFF;color:#000;border:1px solid #666;padding:2px 8px;">white</span> 1.00, 1.00, 1.00</td>
      <td>≥9</td>
      <td>degree 9 and above</td>
      <td>order 9+</td>
    </tr>
  </tbody>
</table>

On algebraic plots, $`o = k = \deg p`$ for the enumerated $`p(z)=\sum_{n=0}^{k} c_n z^n`$ with $`c_k>0`$. The colour is that polynomial's degree. A listed cubic is blue at all of its roots.

On the IIR plot, $`o = M = \deg A`$ for $`A(z)=1+\sum_{m=1}^{M} a_m z^{-m}`$ built from reflection coefficients $`k_0,\ldots,k_{M-1}`$.

Brightness and mark width use different formulas on each plot:

| plot | brightness | mark width |
| --- | --- | --- |
| algebraic blobs | hit count $`w`$ at the same rounded $`(x,y,h,o)`$, splat amplitude $`w \cdot C(o)`$ | blob radius $`r = k_1 k_2^{h-3}`$, lower Brooks $`h`$, wider glow |
| IIR poles | $`0.55 \cdot \mathrm{clip}(\log(1+Q)/6.5,\ 0.06,\ 1) \cdot C(o)`$, higher $`Q`$ when $`|z|`$ is closer to 1 | $`k_1 k_2^{M-2}`$, wider at higher order |

Brooks complexity on the algebraic plots is

$$
h = \sum_{n=0}^{k} \left(|c_n| + 1\right)
$$

IIR pole sharpness is

$$
Q \approx \frac{0.5\,|z|}{1-|z|}
$$

## Algebraic numbers

The default values are $k_1=0.125$ and $k_2=0.5$. Polynomials have leading coefficient $c_k>0$ and degree $k\ge 1$. Each $h$ is a unary bit encoding of $|c_n|$, followed by all sign patterns on the non-leading nonzero coefficients. The CPU path uses `numpy.roots`. The GPU path uses batched float64 companion eigenvalues. Hits at the same $(x,y,h)$ and degree merge into one blob with $`w>1`$.

The viewport is three numbers passed to `render`: the centre $`(ox, oy)`$ in the complex plane, and `zoom` in pixels per unit. The default view is centred on the origin with `zoom = height / 5`.

```bash
# full view (top image)
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --height 2160 \
  --no-labels --png algebraics/algebraics_h17_3840px_default.png

# detail crop like [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png) (second image)
python3 algebraics/algebraics_gpu.py --maxh 17 --width 1920 --height 1080 \
  --ox 0.86 --oy 0.58 --zoom 820 \
  --no-labels --png algebraics/algebraics_h17_wiki_crop.png

# same crop with inline labels (`feat/inline-root-labels`)
python3 algebraics/algebraics_gpu.py --maxh 17 --width 3840 --height 2160 \
  --ox 0.86 --oy 0.58 --zoom 1640 \
  --png algebraics/algebraics_h17_wiki_crop_labeled.png
```

If you change the width, scale `zoom` with the width. At 3840 px wide, the same crop uses `--zoom 1640` ($820 \times 3840/1920$).

The CPU command is:

```bash
python3 algebraics/algebraics.py --maxh 15 --png algebraics/algebraics.png
```

Local PNG and viewer runs label high-hit roots in place (0, ±1, ±1/2, ±i, the sixth roots, φ, …) from the same enumeration. Use `--no-labels` for a splat-only plate.

The Colab notebook is `algebraics/colab_algebraics.ipynb`. Brooks used `maxh=15`. A T4 can run `maxh=17`, and an A100 can run `maxh=18`.

The file `algebraics/algebraics_h17_3840px_default.png` was rendered on a T4 with batch 8192, at 3840×2160, with the default viewport. It has 803744 polynomials, 5744032 roots, and 4660457 unique blobs. GPU root finding took 492 s.

## Constellatio study

Callout map on `master` — yellow leaders and boxes A–E, drawn after the splat by `annotate_constellatio.py`.

![constellatio annotated map, h=17, 7680×4320 (`master`)](algebraics/constellatio_annotated.png)

Inline root labels on `feat/inline-root-labels` — high-hit roots named while the PNG is written (`root_labels.py`).

![algebraic numbers, labelled detail near (0.86, 0.58) (`feat/inline-root-labels`)](algebraics/algebraics_h17_wiki_crop_labeled.png)

The annotated map and the single-colour plots share the same root cache, so `--maxh` must match everywhere.

The annotated map is one labelled full render (`constellatio.pdf`). The single-colour plots are nine full renders, one hue-table entry each (`deg1` to `deg9_plus`).

Callout geometry and labels are in `algebraics/regions/defs.py` (`CALLOUT_REGIONS`). Degree-plot IDs are in the same file (`DEGREE_PLOTS`).

### Annotated map

```bash
python3 algebraics/algebraics_gpu.py --maxh 17 --width 7680 --height 4320 \
  --no-labels --png algebraics/algebraics_h17_7680px.png

python3 algebraics/annotate_constellatio.py algebraics/algebraics_h17_7680px.png
```

### Single-colour degree plots

```bash
python3 algebraics/regions/build.py
# subset:  python3 algebraics/regions/build.py --plots deg3 deg6
# skip PDF: python3 algebraics/regions/build.py --no-pdf
```

`build.py` reads the consolidated blob cache. The default size is 7680×4320, with viewport centre `(0,0)` and `zoom = height/5`. It writes one folder per plot:

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
      <td><span style="background-color:#FF9900;color:#000;padding:2px 8px;">orange</span></td>
    </tr>
    <tr>
      <td><code>deg6</code></td>
      <td>6</td>
      <td><span style="background-color:#00FFFF;color:#000;padding:2px 8px;">cyan</span></td>
    </tr>
    <tr>
      <td><code>deg7</code></td>
      <td>7</td>
      <td><span style="background-color:#FF00FF;color:#fff;padding:2px 8px;">magenta</span></td>
    </tr>
    <tr>
      <td><code>deg8</code></td>
      <td>8</td>
      <td><span style="background-color:#999999;color:#fff;padding:2px 8px;">grey</span></td>
    </tr>
    <tr>
      <td><code>deg9_plus</code></td>
      <td>≥9</td>
      <td><span style="background-color:#FFFFFF;color:#000;border:1px solid #666;padding:2px 8px;">white</span></td>
    </tr>
  </tbody>
</table>

Where many blobs of the same degree overlap, the colour clips to white or gold.

For a one-shot Colab run, use `algebraics/colab_algebraics.ipynb` or `algebraics/run_colab_cli.sh`. That path does the GPU render, the degree plots, and the constellatio PDF, then downloads `constellatio.pdf` and `constellatio_regions.zip`.

## Random stable IIR poles

The script draws reflection coefficients with $`|k|<1`$ and uses the Levinson step-up to build Schur polynomials. Poles of high order pile up near $`|z|=1`$.

```bash
python3 iir-poles/lattice_poles.py --orders 2-16 --n 350
python3 iir-poles/lattice_poles.py --gpu --orders 2-18 --n 1500 --width 3840 --height 2160 --png iir-poles/poles_o18_3840px.png
```

The file `iir-poles/poles_o18_3840px.png` was rendered on a T4 for orders 2 to 18. It has about 197k poles. The median $`|z|`$ is 0.76 at order 2 and 0.999 at order 18. The GPU run takes about 1 minute. The Colab notebook is `iir-poles/colab_poles.ipynb`.

## License

This work is licensed under [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/). The legal code is in `LICENSE`.

You are free to copy, distribute, transmit, and adapt the work, provided you give appropriate credit, link to the license, and say if changes were made. Do not suggest that the licensor endorses you or your use.

The algebraic-number plots adapt Stephen J. Brooks's 2010 sketch ([source](https://en.wikipedia.org/wiki/User:Stephen_J._Brooks/algebraics/src), [Algebraicszoom.png](https://commons.wikimedia.org/wiki/File:Algebraicszoom.png)), also under CC BY 3.0. A copy of his program is `algebraics/original.c`. This repository reimplements that method in Python (CPU and GPU) and adds the annotated map, the degree plots, and the IIR-pole figure. Brooks does not endorse this adaptation.
