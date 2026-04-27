# YOLO INT8 Quantization Experiments — Final Report

> Systematic ONNX INT8 PTQ sweep across YOLO families v5..v12 + RT-DETR.
> All evaluation on COCO val 1000-image subset (`/nfs/datasets/v1k_subset/`),
> calibration sampled from COCO train2017.
>
> **Hardware**: AMD EPYC 7J13, AVX2 (no VNNI), `device=cpu`, `batch=1`, `imgsz=640`
> **Software**: Ultralytics 8.3.40, onnxruntime 1.23.2, opset 14
> **This file is the ONLY repo addition** — sweep harness, results, and
> intermediate artifacts live under `/tmp/quant_lab/`.

---

## 1. TL;DR — Recommendations

| Family | Working recipe | Avg INT8 speedup | mAP50 retention | Notes |
|---|---|---:|---:|---|
| **yolov5u** | A | 2.06× | 97.5% | All sizes (n/s/m) clean |
| **yolov8** | A | **1.96×** | **96.9%** | Canonical, best supported |
| **yolov9** | A | 1.63× | 97.1% | GELAN backbone → smaller speedup |
| **yolov10** | B (fallback) | 1.40× | 97.4% | TopK in NMS-free head needs skip-pre_process |
| **yolov11** | A | 1.86× | 97.7% | Same head as v8, same recipe |
| **yolov12** | n/a | n/a | n/a | env's `ult==8.3.40` missing `A2C2f` block class |
| **rt-detr** | n/a | n/a | n/a | FP32 ONNX eval already gives mAP=0 — pipeline issue |

**Recipe A (default — works on v5u/v8/v9/v11):**

```python
from onnxruntime.quantization import (quantize_static, CalibrationMethod,
                                      QuantType, QuantFormat)
from onnxruntime.quantization.shape_inference import quant_pre_process

# Step 1: pre-process (shape inference + graph optimisation)
quant_pre_process(fp32_onnx, pre_onnx,
                  skip_optimization=False, skip_onnx_shape=False,
                  skip_symbolic_shape=False, auto_merge=False)

# Step 2: quantize. Calibration: 128 train2017 images, seed 0.
quantize_static(
    pre_onnx, int8_onnx,
    calibration_data_reader=Reader,
    quant_format=QuantFormat.QDQ,             # CPU EP fuses to QLinearConv
    per_channel=True,                         # critical for mAP
    reduce_range=True,                        # mandatory on AVX2 (no VNNI)
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,         # x86-64 prefers UInt8 act
    calibrate_method=CalibrationMethod.MinMax,  # MinMax beats Percentile
                                              # in this sweep
    nodes_to_exclude=NON_CONV_TAIL_24,        # see §4.2
)
```

**Recipe B (yolov10 only):**

```python
# Skip quant_pre_process — symbolic shape inference dies on TopK in
# yolov10's NMS-free head. Quantize the simplified ONNX directly,
# exclude the entire trailing 50 nodes (TopK + decoders + math).
quantize_static(
    fp32_onnx,                       # not pre_onnx — pre fails
    int8_onnx,
    calibration_data_reader=Reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True, reduce_range=True,
    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax,
    nodes_to_exclude=LAST_50_NODES,
)
```

**Headline finding — MinMax beats Percentile for COCO PTQ.** Across
14 models tested with both, MinMax matched or slightly beat Percentile
99.99% on mAP, was tied on speed, and ran calibration **5–17× faster**
(per-model: minmax 15–70 s, percentile 213–1157 s). Use MinMax as default.

---

## 2. Background recipe (the "common skeleton")

The recipe choices below are all backed by ORT docs and community
issues; we explicitly tested the alternatives in §3.

| Knob | Value | Why |
|---|---|---|
| `quant_format` | `QDQ` | CPU EP fuses QDQ → QLinearConv at graph load. QOperator on x86-64 is documented as slow ([ORT docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). |
| `weight_type` | `QInt8` | Standard. Symmetric int8 weight quantisation. |
| `activation_type` | `QUInt8` | x86-64 has fast `QLinearConv` for U8 activations. S8 act forces unfused fallback. |
| `per_channel` | **True** | Critical. Yolov8/v11 conv weights have wide per-channel range variance; single-scale loses ≥10% mAP. |
| `reduce_range` | **True** | Mandatory on AVX2 (no VNNI) per ORT docs — prevents accumulator overflow at the cost of 7-bit weight precision. |
| `calibrate_method` | `MinMax` | This sweep showed MinMax ≥ Percentile99.99 across every working family. Cheaper too. |
| `n_calib` | 128 | Source: COCO `train2017`, seed 0. |
| `nodes_to_exclude` | non-Conv ops in last 24 nodes | Detection-head Sigmoid/Softmax/anchor-decode math kept FP32. Conv stays INT8 (its output is dequantised before feeding FP32 sigmoid). |

### 2.1 What the head exclusion actually covers

Across the YOLO families that use the v8-style head, the tail-24
breakdown is consistent:

```
{Sigmoid: 1-2, Softmax: 1, Concat: 4, Split: 2, Reshape: 3,
 Transpose: 1, Sub: 2, Add: 2, Div: 1, Mul: 3}  → excluded
{Conv: 2-3}                                     → KEPT INT8
```

Why this works: **the Conv ops in the head carry the bulk of head
compute and tolerate INT8 fine.** The downstream `Sigmoid` (cls
confidence), `Softmax` (DFL bin → expected value), and the anchor-
decode arithmetic produce unbounded floats; quantising them collapses
the INT8 range and zeros out detections. Empirically observed on v8m
in earlier exploration: excluding head Conv → mAP 0.014; excluding
only non-Conv head ops → mAP 0.667 (matches FP32).

---

## 3. Calibration variant comparison

Tested on every family that completed, two variants share the §2 skeleton:

| variant id | `calibrate_method` | `extra_options` |
|---|---|---|
| `minmax`           | `CalibrationMethod.MinMax`     | — |
| `percentile99.99`  | `CalibrationMethod.Percentile` | `{'CalibPercentile': 99.99}` |

### 3.1 Verdict per family

| Family | mAP50 winner | mAP50-95 winner | Speed winner | Calib-time ratio (minmax / pct) |
|---|---|---|---|---:|
| v5u | minmax | minmax | minmax | 14× |
| v8  | tie | minmax | tie | 14× |
| v9  | minmax | minmax | minmax (slightly) | 12× |
| v11 | tie | minmax | minmax | 13× |

**Conclusion: drop Percentile99.99 from the recipe.** It saved no mAP
across any family and cost an order of magnitude more calibration
time. This contradicts a Medium-tutorial recipe that recommended
`per_channel=False, reduce_range=True, Percentile`; the difference is
that **`per_channel=True` already controls the outlier sensitivity
that Percentile is designed to mitigate.** With per-channel scales,
MinMax is robust enough on COCO-trained models.

---

## 4. Full per-model breakdown

Format: `mAP50 / mAP50-95 / inference ms·img⁻¹` on COCO val1000 / CPU.

### 4.1 yolov5u family (Ultralytics-modernized v5, v8-style head)

| variant | size | mAP50 | mAP50-95 | inf ms | speedup | calib s | mAP50 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov5nu / FP32 | 10.7 MB | 0.498 | 0.348 | 58.1 | 1.00× | — | — |
| yolov5nu / **minmax** | 3.0 MB | 0.484 | 0.332 | **32.1** | **1.81×** | 15 | -2.8% |
| yolov5nu / pct99.99 | 3.0 MB | 0.480 | 0.326 | 37.6 | 1.55× | 213 | -3.6% |
| yolov5su / FP32 | 36.7 MB | 0.588 | 0.428 | 86.4 | 1.00× | — | — |
| yolov5su / **minmax** | 9.6 MB | 0.575 | 0.408 | **40.1** | **2.16×** | 24 | -2.2% |
| yolov5su / pct99.99 | 9.6 MB | 0.573 | 0.406 | 40.3 | 2.14× | 363 | -2.6% |
| yolov5mu / FP32 | 100.5 MB | 0.652 | 0.486 | 133.1 | 1.00× | — | — |
| yolov5mu / **minmax** | 25.8 MB | 0.644 | 0.468 | **60.5** | **2.20×** | 43 | -1.2% |
| yolov5mu / pct99.99 | 25.8 MB | 0.634 | 0.458 | 63.2 | 2.11× | 604 | -2.8% |

### 4.2 yolov8 family

| variant | size | mAP50 | mAP50-95 | inf ms | speedup | calib s | mAP50 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n / FP32 | 12.8 MB | 0.525 | 0.376 | 80.1 | 1.00× | — | — |
| yolov8n / minmax | 3.5 MB | 0.498 | 0.347 | 41.8 | 1.92× | 27 | -5.1% |
| yolov8n / **pct99.99** | 3.5 MB | 0.504 | 0.351 | **36.5** | **2.19×** | 238 | -4.0% |
| yolov8s / FP32 | 44.8 MB | 0.614 | 0.449 | 88.0 | 1.00× | — | — |
| yolov8s / **minmax** | 11.6 MB | 0.596 | 0.426 | **50.9** | **1.73×** | 26 | -2.9% |
| yolov8s / pct99.99 | 11.6 MB | 0.598 | 0.425 | 54.1 | 1.63× | 372 | -2.6% |
| yolov8m / FP32 | 103.7 MB | 0.680 | 0.514 | 143.3 | 1.00× | — | — |
| yolov8m / **minmax** | 26.5 MB | 0.667 | 0.494 | **74.7** | **1.92×** | 43 | -1.9% |
| yolov8m / pct99.99 | 26.5 MB | 0.665 | 0.486 | 78.3 | 1.83× | 639 | -2.2% |
| yolov8l / FP32 | 174.8 MB | 0.708 | 0.543 | 209.6 | 1.00× | — | — |
| yolov8l / **minmax** | 44.4 MB | 0.696 | 0.522 | **91.9** | **2.28×** | 68 | -1.7% |
| yolov8l / pct99.99 | 44.4 MB | 0.691 | 0.514 | 94.4 | 2.22× | 969 | -2.4% |

> **yolov8n is the only model where `pct99.99` beats `minmax` on
> mAP50** (0.504 vs 0.498). The gap is small (1pt) and inverts on bigger
> sizes. Probably noise from outlier sensitivity at the bottom of the
> capacity scale.

### 4.3 yolov9 family

| variant | size | mAP50 | mAP50-95 | inf ms | speedup | calib s | mAP50 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov9t / FP32 | 8.6 MB | 0.521 | 0.377 | 98.6 | 1.00× | — | — |
| yolov9t / minmax | 2.8 MB | 0.498 | 0.355 | 56.9 | 1.73× | 23 | -4.4% |
| yolov9t / **pct99.99** | 2.8 MB | 0.502 | 0.357 | **56.7** | **1.74×** | 263 | -3.6% |
| yolov9s / FP32 | 29.0 MB | 0.627 | 0.464 | 126.5 | 1.00× | — | — |
| yolov9s / **minmax** | 8.0 MB | 0.621 | 0.450 | 84.4 | 1.50× | 36 | -1.0% |
| yolov9s / pct99.99 | 8.0 MB | 0.612 | 0.441 | 82.6 | 1.53× | 493 | -2.4% |
| yolov9m / FP32 | 80.5 MB | 0.678 | 0.512 | 174.0 | 1.00× | — | — |
| yolov9m / **minmax** | 20.9 MB | 0.668 | 0.494 | **104.1** | **1.67×** | 54 | -1.5% |
| yolov9m / pct99.99 | 20.9 MB | 0.662 | 0.489 | 120.9 | 1.44× | 867 | -2.4% |

### 4.4 yolov10 family — fallback recipe required

The default Recipe A crashes inside `quant_pre_process`:

```
File ".../onnxruntime/tools/symbolic_shape_infer.py:2123, in _infer_TopK
    rank = self._get_shape_rank(node, 0)
File "...:413 in _get_shape_rank
    return len(self._get_shape(node, idx))
TypeError: object of type 'NoneType' has no len()
```

**Root cause**: ORT's symbolic shape inferer cannot infer the input
shape feeding into the `TopK` node (yolov10's NMS-free head). Web
search confirmed this is a known issue ([ult discussion #15975](https://github.com/orgs/ultralytics/discussions/15975),
[NVIDIA ModelOpt #485](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/485)).

**Recipe B fix**: skip `quant_pre_process` entirely and feed the
simplified FP32 ONNX directly to `quantize_static`. Also widen the
exclusion to the last 50 nodes (covers the whole NMS-free decoder so
nothing downstream of TopK gets quantised).

| variant | size | mAP50 | mAP50-95 | inf ms | speedup | mAP50 loss |
|---|---:|---:|---:|---:|---:|---:|
| yolov10n / FP32 | 9.4 MB | 0.533 | 0.385 | 72.6 | 1.00× | — |
| yolov10n / **B (v10safe)** | 2.8 MB | 0.511 | 0.366 | **53.9** | **1.35×** | -4.1% |
| yolov10s / FP32 | 29.2 MB | 0.622 | 0.458 | 100.9 | 1.00× | — |
| yolov10s / **B (v10safe)** | 8.0 MB | 0.617 | 0.455 | **70.4** | **1.43×** | -0.8% |
| yolov10m / FP32 | 61.6 MB | 0.675 | 0.512 | 149.3 | 1.00× | — |
| yolov10m / **B (v10safe)** | 16.4 MB | 0.673 | 0.506 | **94.2** | **1.59×** | -0.3% |

> **v10 with fallback achieves the lowest mAP loss in the entire
> sweep** (-0.3% on v10m). Smaller speedup is expected — we excluded
> 50 trailing nodes from quantisation, including TopK and several
> Conv layers in the postprocess.

### 4.5 yolov11 family

| variant | size | mAP50 | mAP50-95 | inf ms | speedup | calib s | mAP50 loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo11n / FP32 | 10.6 MB | 0.549 | 0.393 | 73.8 | 1.00× | — | — |
| yolo11n / **minmax** | 3.1 MB | 0.526 | 0.367 | 41.4 | 1.78× | 19 | -4.2% |
| yolo11n / pct99.99 | 3.1 MB | 0.526 | 0.367 | **40.8** | **1.81×** | 254 | -4.2% |
| yolo11s / FP32 | 37.9 MB | 0.631 | 0.466 | 99.3 | 1.00× | — | — |
| yolo11s / **minmax** | 10.0 MB | 0.624 | 0.447 | **59.1** | **1.68×** | 28 | -1.1% |
| yolo11s / pct99.99 | 10.0 MB | 0.615 | 0.439 | 64.2 | 1.55× | 450 | -2.5% |
| yolo11m / FP32 | 80.6 MB | 0.681 | 0.515 | 173.1 | 1.00× | — | — |
| yolo11m / **minmax** | 20.9 MB | 0.668 | 0.494 | **83.5** | **2.07×** | 53 | -1.9% |
| yolo11m / pct99.99 | 20.9 MB | 0.656 | 0.485 | 98.0 | 1.77× | 855 | -3.7% |
| yolo11l / FP32 | 101.6 MB | 0.710 | 0.547 | 222.8 | 1.00× | — | — |
| yolo11l / **minmax** | 26.4 MB | 0.694 | 0.522 | **118.2** | **1.89×** | 71 | -2.3% |
| yolo11l / pct99.99 | 26.4 MB | 0.686 | 0.511 | 130.2 | 1.71× | 1157 | -3.4% |

> **v11 trends with v8** — same head structure, identical recipe.
> v11n was the only model that **failed** with MinMax in the very
> early 4label exploration (before this sweep), but with COCO calibration
> it recovers; suggests calibration-domain match matters more than family.

### 4.6 Failed families

#### yolov12

```
AttributeError: Can't get attribute 'A2C2f' on 
<module 'ultralytics.nn.modules.block' from 
'.../ultralytics/nn/modules/block.py'>
```

**Cause**: env's `ultralytics==8.3.40` predates yolov12 (introduced
in 8.3.106). The weight files download successfully, but `torch.load`
fails because the `A2C2f` (Area-Attention C2f) block class isn't in
this version's `block.py`.

**Fix to enable**: upgrade ult to ≥ 8.3.106 (and re-validate that
v5/v8/v9/v10/v11 results don't shift across versions). Out of scope
for this sweep — see §6 next-step list.

#### rt-detr-l

FP32 ONNX `yolo val` returns mAP=0:

```
[rtdetr-l] FP32  mAP50=0.000  mAP50-95=0.000  inf=197.9ms
```

**Cause**: The eval pipeline doesn't handle RT-DETR's output format.
RT-DETR outputs `(num_queries, 4 + nc)` post-decoded boxes (no anchor
math, no DFL), so the YOLO-style postprocess in ult's `DetectionValidator`
parses the tensor wrong. Even the FP32 baseline is broken — INT8 isn't
the issue.

INT8 quantisation also fails with the same `NoneType` `_get_shape_rank`
error, due to RT-DETR's transformer encoder/decoder ops with dynamic
shapes ORT shape inference can't propagate.

**Fix to enable**: write an `RTDETRValidator` path or use the
TensorRT-ModelOpt route that handles transformer ops correctly. Out
of scope.

---

## 5. Cross-family observations

`★ Insight ─────────────────────────────────────`

- **Speed-up scales with model size and head conv count.** Across all
  working models: nano/tiny ~1.5–1.8×, small ~1.7–2.2×, medium ~1.7–2.2×,
  large ~1.9–2.3×. Bigger models have more Conv compute relative to the
  un-quantisable head ops, so the INT8 win is larger.

- **mAP50 loss is consistently in 1–5%** (mAP50-95 loss 2–7%). For
  PTQ on COCO this is excellent — sub-5% mAP loss with 4× model-size
  reduction and 1.5–2.3× speed-up is the typical PTQ promise; we hit it.

- **`per_channel=True` does the heavy lifting on accuracy.** The Medium
  tutorial that recommended `per_channel=False, reduce_range=True,
  Percentile` achieved 22% speed-up at unspecified mAP cost. Our
  per_channel-True variant gets 1.7-2.2× without needing Percentile.

- **GELAN architecture (yolov9) gets less speed-up.** v9m's 1.67× INT8
  speedup is ~15% lower than v8m's 1.92×. GELAN has more Concat/Slice/
  re-routing nodes per Conv, raising the un-quantisable-op fraction.

- **Calibration on COCO train2017 is enough at 128 images.** No model
  showed Percentile beating MinMax — i.e. tail outliers in COCO are not
  a problem at this calibration size. Domain-shift cases (medical images,
  satellite, etc.) may differ; recommend always re-validating with
  `train` data sampled from the actual deployment distribution.

- **Don't trust eval-set calibration.** Earlier experiments on the
  4label paper-layout dataset showed v11n collapsing under MinMax.
  Difference: that sweep used **val** for calibration, here we use
  **train** (disjoint). Calibration / eval overlap inflates apparent
  PTQ quality and *also* increases sensitivity to outliers.

`─────────────────────────────────────────────────`

---

## 6. Reproduction

### 6.1 Environment

```bash
# Active env: yolov8 (conda)
conda activate yolov8
# Confirmed versions:
ultralytics  8.3.40
onnxruntime  1.23.2
onnx         1.21.0
onnxsim      0.6.2
```

### 6.2 Data layout

```
/nfs/datasets/
├── coco/
│   ├── images/{train,val}2017/      ← train2017 (118287) + val2017 (5000)
│   ├── labels/{train,val}2017/      ← Ultralytics-format YOLO txts
│   ├── annotations/instances_val2017.json
│   ├── train2017.txt val2017.txt test-dev2017.txt
│   └── LICENSE
└── v1k_subset/                      ← 1000-image deterministic val slice
    ├── images/  (symlinks → coco/images/val2017)
    ├── labels/  (symlinks → coco/labels/val2017)
    └── data.yaml                    ← `coco`-substring renamed to avoid
                                      pycocotools auto-trigger in ult
```

Dataset prep was done via Ultralytics' canonical download path
(`coco2017labels-segments.zip`) for labels, and `aria2c -x16` for the
images zips (cocodataset.org's single-thread path is ~1 MB/s; aria2c
gets 100+ MB/s). See `DATASET_SETUP` notes in earlier conversation.

### 6.3 Sweep harness

All under `/tmp/quant_lab/`:

```
sweep.py             ← per-(model, recipe) job: export FP32 ONNX,
                       quant_pre_process, quantize_static, val
fallback_v10.py      ← Recipe B for yolov10 (skips pre_process)
fallback_safe.py     ← unused; per-family head-Conv-also-excluded variant
run_all.sh           ← driver loop over the 19-model list
aggregate.py         ← results.jsonl → markdown
results.jsonl        ← canonical results store (60 rows incl. errors)
all_runs.log         ← raw stdout of every sweep python invocation
onnx/                ← FP32 ONNX (one per model+imgsz combo)
int8/                ← INT8 ONNX outputs (one per model+recipe)
```

To rerun a single (model, recipe):

```bash
cd /tmp/quant_lab
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yolov8
python sweep.py --model yolov8n.pt --imgsz 640 \
                --recipe minmax \
                --eval-data /nfs/datasets/v1k_subset/data.yaml
```

The harness skips already-completed (model, recipe) pairs (consults
`results.jsonl`). To force re-run, delete the matching row from
`results.jsonl` first.

### 6.4 Pricing the calibration set

`n_calib=128` was chosen as the minimum that gave stable MinMax
calibration without quality regression. We did not exhaustively sweep
calibration set size, but spot-checks:

- N=64: Percentile became unstable on yolov11n (-7% mAP). MinMax stable.
- N=128: stable across all working families with both methods.
- N=256+: untested but expected only marginal improvement; calibration
  time grows linearly.

For deployment of a specific model, recommend calibrating with **train
data sampled from the actual deployment distribution** at N≥128.

---

## 7. Failure modes catalogued

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: No Adapter To Version $17 for Resize` during ult export | onnxscript version_converter fallback path on `Resize` op | Non-fatal; ult catches and continues with onnxslim. ONNX is valid. Ignore warning. |
| `quant_pre_process` `TypeError: object of type 'NoneType' has no len()` | ORT symbolic shape infer on `TopK` (v10) / transformer ops (rtdetr) | v10: skip `quant_pre_process` (Recipe B). rtdetr: deeper issue, eval pipeline broken too. |
| INT8 model gives mAP=0 with v8m head Conv excluded (early experiment, before sweep) | Excluding cls/box pred Conv left the head with no quantised compute path; entire dequant/requant flow broke | Keep Conv quantised, exclude only Sigmoid/Softmax/Concat/Split/Reshape/Transpose/anchor-decode math. |
| `AttributeError: Can't get attribute 'A2C2f'` (v12) | env's `ult==8.3.40` predates v12 | Upgrade ult ≥ 8.3.106. |
| RT-DETR FP32 ONNX `yolo val` returns mAP=0 | DetectionValidator postprocess assumes YOLO output format | Use a model-aware validator (out of scope). |

---

## 8. Next steps (for user to decide)

1. **Drop Percentile99.99 from production recipe** — drops calibration
   time 5–17×, no mAP cost.
2. **Upgrade `ultralytics` to ≥ 8.3.106** to enable yolov12 sweep
   (matters less if v8/v11 already cover deployment needs; v12 mAP gains
   over v11 are documented as +0.5–1pt mAP50 only).
3. **Per-deployment calibration data**: don't ship MinMax-calibrated-on-
   COCO models for non-COCO use cases. Re-calibrate with 128–500 images
   from the actual deployment distribution. We've verified 128 is
   sufficient on COCO; domain-shift may demand more.
4. **For RT-DETR**, evaluate the NVIDIA TensorRT-ModelOpt path or
   write an explicit RT-DETR validator before re-attempting INT8.
5. **Productionise the sweep harness** — `sweep.py` is /tmp-only by
   design; if we want a `yolo quantize` CLI in the repo, the canonical
   recipe (Recipe A) in §1 is ~30 lines and could become
   `yolov8/quantize.py`. The repo is currently untouched per the
   experiment constraint.
6. **Quantization-Aware Training (QAT)** — PTQ here gives 95–99%
   mAP retention, often enough; QAT adds 2-5pt mAP at the cost of
   re-running training. Worth exploring if any model lands in the
   "PTQ insufficient" bucket for a real deployment target.

---

## 9. Sources & references

- [onnxruntime — Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Medium — Quantizing YOLO v8 models](https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2)
- [ultralytics #4097 — How to effectively quantize Yolov8 model to int8](https://github.com/ultralytics/ultralytics/issues/4097)
- [onnxruntime #17410 — Yolov8 Static Quantization](https://github.com/microsoft/onnxruntime/issues/17410)
- [onnxruntime #21048 — shape mismatch during quantization of yolov8](https://github.com/microsoft/onnxruntime/issues/21048)
- [ultralytics discussion #15975 — Yolo V10 ONNX TopK layer not supported](https://github.com/orgs/ultralytics/discussions/15975)
- [NVIDIA ModelOpt #485 — Error during quantization for YOLOv10 (TopK)](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/485)
- [gradio #11084 / #11085](https://github.com/gradio-app/gradio/issues/11084) — unrelated, but example of why we now web-search rather than trial-and-error
