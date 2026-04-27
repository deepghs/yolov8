# CLAUDE.md

> **Note for agents**: `AGENTS.md` at the repository root is a symbolic link
> pointing to this file (`AGENTS.md -> CLAUDE.md`). The two are byte-identical
> on purpose. **Edit only `CLAUDE.md`.** Do not replace `AGENTS.md` with a
> standalone file, or the two guides will drift. This single file serves
> Claude Code, Cursor, Codex, and any other agent that follows the
> `AGENTS.md` convention.

---

## 1. Repository overview

- **PyPI package**: `dghs-yolov8`
- **Import name**: `yolov8` (the package directory `yolov8/`)
- **Purpose**: a thin wrapper around
  [Ultralytics](https://github.com/ultralytics/ultralytics) detection /
  segmentation models (YOLOv8/v9/v10/v11/v12, RT-DETR). It standardises the
  training entry point, ONNX export, publishing to HuggingFace / Roboflow,
  and aggregating already-published models into a README table.
- **Upstream**: `git@github.com:deepghs/yolov8.git` (organisation `deepghs`).
- **License**: Apache-2.0 (see `LICENSE`).
- **Python**: `>=3.7`.
- **Metadata source of truth**: `yolov8/config/meta.py` (`__VERSION__`, etc.).
  `setup.py` `exec`s this file at import time to read the version, so bump
  the version there.

### 1.1 Directory layout

```
.
├── README.md                  user-facing docs
├── CLAUDE.md / AGENTS.md      this file / symlink (agent-facing)
├── LICENSE                    Apache-2.0
├── Makefile                   docs / unittest entry points (see caveat below)
├── MANIFEST.in
├── setup.py                   wraps setuptools via distutils.core.setup
├── requirements.txt           main runtime deps (ultralytics<=8.3.105; supports v8/v9/v10/v11/v12/rtdetr)
├── requirements-roboflow.txt  optional: roboflow publish path (pins ultralytics==8.0.196)
├── requirements-onnx.txt      onnx / onnxruntime / onnxsim / onnxoptimizer
├── requirements-doc.txt       doc deps
├── requirements-test.txt      test deps
├── cloc.sh                    line-count helper (tooling only)
└── yolov8/                    actual Python package
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   └── meta.py            version / author / description constants
    ├── train/
    │   ├── __init__.py        re-exports train_object_detection / train_segmentation
    │   ├── object_detection.py    detection training entry
    │   └── segmentation.py        segmentation training entry
    ├── export.py              exports best.pt + ONNX + curves/csv from a workdir, with anonymisation
    ├── publish.py             `python -m yolov8.publish {huggingface,roboflow}` CLI
    ├── list.py                `python -m yolov8.list` scans an HF repo and writes a README table
    ├── onnx.py                ONNX export (YOLO/RTDETR); also dual-head
    │                          (predictions + embedding) export — see §1.7
    ├── embed.py               universal image-embedding extractor for any
    │                          ult model object (YOLO/RTDETR/raw BaseModel);
    │                          internal — not re-exported from __init__.py
    └── utils/
        ├── __init__.py        re-exports the utilities below
        ├── ckpt.py            derive_model_meta — (model_type, problem_type) from a checkpoint
        ├── cli.py             GLOBAL_CONTEXT_SETTINGS, print_version
        ├── md.py              markdown table -> DataFrame
        ├── pe.py              pretty-print numbers as k/M/G
        └── threshold.py       compute_threshold_data — F1 / threshold from validator metrics
```

### 1.2 Core data flow

1. **Train**: `yolov8.train.train_object_detection` / `train_segmentation`.
   - Picks the model family (v8/9/10/11/12 use `YOLO`, `rtdetr` uses
     `RTDETR`), conventionally writes to `runs/<task_name>/`. **No
     local sidecar metadata is written**; `(model_type, problem_type)`
     are derived on demand from the trained checkpoint via
     `yolov8.utils.derive_model_meta` (see §1.6).
   - If `weights/last.pt` exists, training auto-resumes (`resume=True`).
2. **Export**: `yolov8.export.export_model_from_workdir(workdir, export_dir, ...)`
   - When copying `weights/best.pt`, **`train_args.data` / `project` / `model`
     are sha3-anonymised** to avoid leaking the trainer's local paths into
     published artifacts. Do not revert this to plaintext.
   - `threshold.json` is written **at training time** by
     `yolov8.utils.compute_threshold_data` directly from the validator's
     in-memory `f1_curve` / `px` arrays (mean across classes →
     `f1_score`/`threshold`, plus a per-class breakdown). Export just
     ships the file if it exists. The previous OCR-from-F1_curve.png
     path has been removed.
   - `yolov8.onnx.export_yolo_to_onnx` defaults to
     `dynamic=True, simplify=True, opset=14`.
3. **Publish**: `yolov8.publish`
   - `huggingface`: needs `HF_TOKEN`; uploads under `<name>/...` in the HF
     model repo. This is the primary publish path.
   - `roboflow`: needs `ROBOFLOW_APIKEY`; **only yolov8 models are
     supported**, and the `roboflow` package is an optional extra
     (`requirements-roboflow.txt`). The import is therefore lazy inside
     the `roboflow` subcommand — do not move it back to module top.
4. **Aggregate README**: `yolov8.list`
   - Scans `*/model.pt` in an HF repo, recomputes FLOPS / params, reads
     `threshold.json` / `model_type.json`, builds a unified table, and
     replaces the existing table in `README.md` in-place (detection
     condition: header contains `Model / FLOPS / Params / Labels`).

### 1.3 Command reference

| Purpose | Command |
| --- | --- |
| Train (library API) | `from yolov8.train import train_object_detection, train_segmentation` |
| Local zip export | `python -m yolov8.export -w runs/<task>` |
| Publish to HuggingFace | `HF_TOKEN=... python -m yolov8.publish huggingface -w runs/<task> -r <user>/<repo>` |
| Publish to Roboflow | `ROBOFLOW_APIKEY=... python -m yolov8.publish roboflow -w runs/<task> -p <ws>/<proj> -v <ver>` |
| Refresh HF README table | `python -m yolov8.list -r <user>/<repo>` |
| Unit tests | `make unittest` (or `pytest test/...` directly) |
| Build docs | `make docs` / `make pdocs` (needs `requirements-doc.txt`) |
| Line count | `./cloc.sh [--loc|--comments|--percentage]` |

> ⚠️ **Makefile caveat**: `SRC_DIR := ./imgutils` is **leftover from a
> template** and is wrong for this repo (the package is `yolov8`).
> `make unittest` will point coverage at a path that does not exist.
> Either fix the line first, or invoke `pytest` directly when running
> tests. Verify nothing upstream still depends on the old path before
> changing it.

### 1.4 Choosing dependency sets

- `requirements.txt` is the **single main dependency set** and constrains
  `ultralytics<=8.3.105`, so all of v8 / v9 / v10 / v11 / v12 / rtdetr
  train out of the box. There is no longer a `requirements-raw.txt`
  (it was merged into `requirements.txt`).
- `requirements-onnx.txt` is required for ONNX export (`yolov8/onnx.py`,
  `yolov8/export.py`); install it alongside `requirements.txt`.
- `requirements-roboflow.txt` is an **optional** extra for the legacy
  Roboflow publish path. It pulls in `roboflow>=1.0.1` and pins
  `ultralytics==8.0.196` (the version the Roboflow SDK's
  `version.deploy("yolov8", ...)` is most compatible with). Installing
  this extra therefore downgrades ultralytics, which is the documented
  trade-off. Roboflow is no longer a primary feature; do not promote it
  back into the main set.
- This repo is **not** published to PyPI, and there are no plans to
  publish it. Always install dependencies by pointing pip at the
  in-tree files (`pip install -r requirements.txt`,
  `pip install -r requirements-roboflow.txt`, etc.). Do **not** instruct
  users to run `pip install dghs-yolov8` or `pip install dghs-yolov8[...]`,
  even though `setup.py` happens to collect `requirements-<group>.txt`
  into `extras_require` — that machinery is currently unused.
  The name `zoo` is blacklisted in `setup.py`.

### 1.5 README vs this file

- `README.md` is for human users: how to install, train, and publish.
- `CLAUDE.md` / `AGENTS.md` is for agents: architectural rationale,
  hidden constraints, operational discipline. Sync usage-related
  changes to `README.md`; keep architecture / pitfall / discipline
  changes here.

### 1.7 Image embeddings & dual-head ONNX (internal)

`yolov8/embed.py` and the two ONNX export functions in `yolov8/onnx.py`
(`export_yolo_to_onnx` for predict-only, `export_yolo_to_onnx_with_embedding`
for the dual-head form) together expose a way to obtain a 1D image
embedding from any Ultralytics-supported model object (YOLO/RTDETR/raw
`BaseModel`) and to ship ONNX graphs that downstream consumers like
`deepghs/imgutils` can drop in on. These are deliberately **not**
re-exported from `yolov8/__init__.py` — treat them as an internal API
while we evaluate the surface.

ONNX naming contract (verified against the default
`YOLO.export(format='onnx')` on ultralytics 8.0.196 → 8.4.41 and used
verbatim by `imgutils.generic.yolo` / `imgutils.generic.yoloseg`):

- input  name: `images`, shape `[batch, 3, H, W]`
- head outputs:
  - detect / pose / obb / classify / rtdetr → `output0`
  - segment → `output0` (boxes + mask coeffs) + `output1` (mask prototypes)
- dual-head form appends `embedding` as the trailing output. The
  `output0` (and `output1` when present) tensor is byte-equivalent to
  what the predict-only export produces — exercised by
  `test/test_embed.py::TestPredictParity`.

Metadata. Both exports preserve every `metadata_props` key the upstream
ultralytics exporter writes (`description` / `author` / `date` /
`version` / `license` / `docs` / `stride` / `task` / `batch` / `imgsz`
/ `names`) and add a `dghs.yolov8.*` namespace recording the package
version, the exporter used, the input/output naming, and — for
dual-head — the embedding's layer indices and dimension. When the
source is a training workdir that contains a `threshold.json`, that
file is embedded under `dghs.yolov8.threshold` so the resulting `.onnx`
is fully self-describing. Callers may inject additional pairs via
`extra_metadata=...`.

Hidden constraints worth knowing before editing:

- The extractor must NOT call `inner.predict(x, embed=...)`. The `embed=`
  keyword was added in ultralytics 8.1.x; we still support 8.0.196 (the
  pinned version of the legacy roboflow publish path). Instead, always
  route through `EmbedHead`, which re-implements the embed branch of
  `BaseModel._predict_once` against the version-stable
  `m.i / m.f / save` ModuleList protocol.
- The dual-head wrapper walks `inner.model[:-1]` and explicitly invokes
  the head with the right signature (`RTDETRDecoder.forward(x, batch=None)`
  vs the YOLO `Detect.forward(x)` family). Setting `m.export = True` on
  the head is what makes the ONNX graph emit a single tensor instead of
  the `(decoded, raw)` tuple eval-mode normally returns — mirror this if
  you add new export paths.
- `export_yolo_to_onnx` writes directly to the user-given path via
  `torch.onnx.export`; do **not** revert to the old
  `model.export(format='onnx') + copy()` recipe — it depends on
  ultralytics' internal "export next to ckpt_path" behaviour, which has
  shifted across releases.
- `_attach_metadata` runs **after** `_maybe_simplify`, because some
  onnxsim versions strip `metadata_props` during graph rewriting.
  Don't reorder.
- The compatibility matrix is verified against ultralytics 8.0.196,
  8.1.47, 8.2.103, 8.3.40, 8.3.105, 8.3.253 and 8.4.41 via
  `tmp_embed/compat_smoke.py` in a sibling conda env (`yolov8-compat`).
  If you change either module in a way that touches ultralytics
  internals, re-run the matrix in that separate env — do **not** test
  in the primary `yolov8` env, which has the latest ultralytics pinned
  and would mask 8.0.x regressions. Note that on ult <8.1 the
  underlying `Metric.update` does not store `f1_curve`/`px`, so
  `threshold.json` extraction is structurally impossible there; the
  smoke gracefully reports SKIP for that case.

### 1.8 Threshold capture across early-stop / DDP

`compute_threshold_data` reads `model.trainer.validator.metrics`. That
attribute is populated by ultralytics' final-validation step *only* in
the trainer process. Two situations break the post-`model.train()`
extraction:

* **Patience exhaustion / time-budget early-stop.** ultralytics breaks
  the training loop early, then runs final-validation on `best.pt` and
  fires `on_train_end`. The `model.train()` call still returns and the
  metrics are present — but only because we keep the wrapper alive long
  enough to read them.
* **Multi-GPU / DDP.** `BaseTrainer.train()` calls
  `subprocess.run(generate_ddp_command(...))`. The actual training
  happens in a child process; the parent's `model.trainer.validator.metrics`
  was instantiated *before* the child started and is never repopulated.
  Post-`model.train()` extraction in the parent silently no-ops.

Mitigation: `train_object_detection` / `train_segmentation` register an
`on_train_end` callback (see `yolov8/train/_threshold_callback.py`)
that calls `compute_threshold_data_from_trainer(trainer)` and writes
`threshold.json` from inside the trainer process, so the file lands on
disk regardless of how training terminates. The post-train block is
kept as a safety net but only acts if `threshold.json` doesn't already
exist (i.e. the callback didn't fire). Do **not** remove either path.

The :func:`compute_threshold_data` family also has a third entry point —
`compute_threshold_data_from_validator_stats` — that recomputes the F1
curve from the validator's accumulated `stats` list. ultralytics <8.1
computes the curve inside `ap_per_class` and discards it before
returning, so on those versions the modern attribute path
(`metric.box.f1_curve` / `.px`) returns empty. The
`*_from_validator_stats` fallback gives us the same curve numerically
without touching ultralytics internals further.

When *both* paths fail (no metrics object **and** no usable stats),
the helper logs a single descriptive `WARNING` that names the
installed ultralytics version and points at upgrading to >=8.1; this
is the only situation in which a training-time threshold capture can
silently fail.

### 1.9 Docstring convention (hard rule for `yolov8/`)

Every public function, class, and method under `yolov8/*` uses
**reStructuredText** docstrings with the field-list style:

```rst
"""One-line summary.

Optional longer prose, blank-line-separated paragraphs.

:param name: ...
:type name: type
:param other: ...
:type other: type
:returns: ...
:rtype: type
:raises SomeError: ...

Example::

    >>> from yolov8.foo import bar
    >>> bar(...)
    expected_repr
"""
```

Hard rules:

- ASCII-only (no CJK, no smart quotes).
- Use `:param x:` / `:type x:` / `:returns:` / `:rtype:` /
  `:raises:`. Don't mix in NumPy-style `Parameters` headers or
  Google-style `Args:` blocks.
- Every public symbol gets an `Example::` block with one or more
  `>>> ` doctest lines that demonstrate the API surface; the example
  doesn't have to be doctest-runnable end-to-end (network, big
  weights), but the call must be syntactically valid.
- Private helpers (leading underscore) may keep a one-paragraph
  summary; the field list is optional there.
- Module-level docstrings describe the module's purpose and link to
  the public symbols with `:func:` / `:class:` / `:mod:` cross-refs
  where it helps.

When you add or refactor a function under `yolov8/`, fix its docstring
to match this convention in the same patch — don't ship a public API
without one.

---

## 2. Agent operating rules

### 2.1 Language policy (hard constraint)

This is a hard rule for **everything that is committed to or generated for
the repository itself**:

- **All in-repo text MUST be English.** This applies to:
  source code (identifiers, strings, log messages), code comments,
  docstrings, module-level documentation, Markdown files (including this
  file and `README.md`), `Makefile` / shell-script comments, type stubs,
  YAML / JSON keys & comments, error messages, configuration files,
  generated files that are checked in, **commit messages**, **PR titles
  and descriptions**, **issue replies**, and **release notes**.
- No CJK, no Cyrillic, no other scripts, no inline translations.
  If the user dictates content in another language, translate it into
  English before writing it to a file or sending it to GitHub.
- Identifiers must be ASCII; do not introduce non-ASCII variable, function,
  class, file, or branch names.
- Exceptions: pure data assets that are intrinsically non-English by
  nature (e.g. a dataset of foreign-language labels). When in doubt,
  ask the user.

**Conversational replies in the terminal are a separate channel and do
NOT follow the above rule.** When talking back to the user inside the
agent's chat / terminal session, **mirror the user's language** —
if the user writes Chinese, reply in Chinese; if English, reply in
English; etc. The English-only rule kicks in only when the agent is
about to write to a file in this repo, run `git commit`, or push
content to GitHub via `gh`.

### 2.2 Editing conventions

- **`AGENTS.md` MUST remain a symlink to `CLAUDE.md`.** If a tool /
  sandbox materialised it as a regular file, fold any new content back
  into `CLAUDE.md` and recreate the link with `ln -sf CLAUDE.md AGENTS.md`.
- The training entry points must **not** write a `model_type.json`
  sidecar into the workdir. The previous code did this via a 30-second
  `threading.Timer` daemon (the comment claimed Ultralytics wipes the
  workdir at training start, which is true) — but the timer silently
  loses the file on fast trainings (<30 s), and the same information
  lives unconditionally inside the embedded model object of every
  Ultralytics checkpoint. Use `yolov8.utils.derive_model_meta` /
  `derive_model_meta_from_path` to recover `(model_type, problem_type)`
  from `weights/best.pt` (or `last.pt`) at export / publish / list
  time. Do not reintroduce a sidecar write — the on-HF artifact
  layout is preserved by synthesising `model_type.json` into
  `export_dir` only inside `export.py`.
- Preserve the `train_args` sha3 anonymisation in `export.py`; it is a
  publish-time privacy requirement.
- Bump the version only in `yolov8/config/meta.py`; `setup.py` reads it.
- A new `requirements-<group>.txt` is auto-collected as a
  `setuptools` extra by `setup.py`. The name `zoo` is blacklisted there.

### 2.3 Tests and runtime

- Before running the suite, check whether the `Makefile`'s `SRC_DIR` has
  been fixed; if not, call `pytest` directly, e.g.
  `pytest test -sv -m unittest --cov=yolov8`.
- Real training requires a GPU; do not attempt training in CI. PRs only
  need to verify imports and CLI surfaces.

### 2.4 Git / GitHub identity discipline (hard constraint)

The host machine usually has multiple `gh` logins (`gh auth status` will
list more than one account). **Never assume the current active account
is the one this repo should use**, and never infer the identity from
`~/.gitconfig` or environment. The repo's identity is determined
**only** by this repo's local `git config`.

#### 2.4.1 Identity resolution (dynamic — do not hard-code)

This file deliberately does **not** record "this repo belongs to account
XYZ". Each agent invocation must re-resolve, like so:

1. Pin the repository root to a variable so `cwd` cannot mislead later
   commands:

   ```bash
   REPO_ROOT="$(git rev-parse --show-toplevel)"
   ```

2. Query **this repo's** local config explicitly with `git -C "$REPO_ROOT"`
   and `--local`. Do **not** use a bare `git config user.name`: if the
   shell `cwd` is not inside the repo, or if local config is missing,
   that command silently falls back to global / system layers and
   returns the wrong identity.

   ```bash
   GIT_USER="$(git -C "$REPO_ROOT" config --local user.name)"
   GIT_EMAIL="$(git -C "$REPO_ROOT" config --local user.email)"
   ```

3. Treat `GIT_USER` as the **GitHub login** and ask `gh` for that user's
   token. If `gh` does not have it, **refuse** — do not fall back to any
   other account:

   ```bash
   GH_TOKEN_VAL="$(gh auth token --user "$GIT_USER" 2>/dev/null)" || true
   [ -n "$GH_TOKEN_VAL" ] || {
     echo "refuse gh: no token in gh auth for '$GIT_USER' (from $REPO_ROOT/.git/config)"
     exit 1
   }
   ```

4. Reverse-verify: call `gh api user` with that token and confirm the
   reported login equals `GIT_USER`. Mismatch → refuse:

   ```bash
   GH_LOGIN="$(GH_TOKEN="$GH_TOKEN_VAL" gh api user --jq .login 2>/dev/null)"
   [ "$GH_LOGIN" = "$GIT_USER" ] || {
     echo "refuse gh: token's login '$GH_LOGIN' != local git user.name '$GIT_USER'"
     exit 1
   }
   ```

#### 2.4.2 The only acceptable way to call `gh`

> Do **not** use `gh auth switch`: it flips the global active account,
> contaminating other terminals, other repositories, and concurrent
> tasks. Always inject the token per command instead:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
GH_TOKEN=$(gh auth token --user "$(git -C "$REPO_ROOT" config --local user.name)") \
  gh <subcommand> ...
```

For non-trivial scripts, run the full 4-step check before the real `gh`
call:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
GIT_USER="$(git -C "$REPO_ROOT" config --local user.name)"
[ -n "$GIT_USER" ] || { echo "refuse: $REPO_ROOT has no local user.name"; exit 1; }

GH_TOKEN_VAL="$(gh auth token --user "$GIT_USER" 2>/dev/null)" || true
[ -n "$GH_TOKEN_VAL" ] || { echo "refuse: no gh token for '$GIT_USER'"; exit 1; }

GH_LOGIN="$(GH_TOKEN="$GH_TOKEN_VAL" gh api user --jq .login 2>/dev/null)"
[ "$GH_LOGIN" = "$GIT_USER" ] || {
  echo "refuse: token login '$GH_LOGIN' != local git user '$GIT_USER'"; exit 1;
}

# real command goes here
GH_TOKEN="$GH_TOKEN_VAL" gh <subcommand> ...
```

#### 2.4.3 When the agent MUST refuse to run `gh`

If any of the following holds, stop and surface the reason to the user.
Do not "pick a working account and continue":

1. `git rev-parse --show-toplevel` fails (not inside a git repo).
2. `git -C "$REPO_ROOT" config --local user.name` is empty (no explicit
   local identity).
3. `gh auth token --user "$GIT_USER"` returns nothing (gh has no such
   login).
4. `gh api user --jq .login` (using that token) reports a login that
   differs from `GIT_USER`.
5. The user asks to use a GitHub account that disagrees with the local
   `user.name`. Pause and confirm with the user; do not silently swap.

#### 2.4.4 `git push` and other push-side operations

- `git push` goes over SSH and is governed by `~/.ssh/config` host
  aliases on the user's machine. It does not need `gh` to be involved.
- The moment something needs `gh pr create` / `gh pr merge` /
  `gh release` / `gh api` / etc., go through the token-injection form
  in §2.4.2. Never let `gh` use the default active account.

#### 2.4.5 Don'ts

- ❌ `gh auth switch ...` (mutates global active account)
- ❌ `gh auth login` (unless the user explicitly asks to add an account)
- ❌ Editing `~/.config/gh/hosts.yml`
- ❌ Changing this repo's `.git/config` `user.name` / `user.email` to
   "make `gh` work"
- ❌ Using `--global` / `--system` to override identity
- ❌ Bare `git config user.name` (without `-C "$REPO_ROOT" --local` —
   results are unreliable)
- ❌ Bare `gh <cmd>` (without `GH_TOKEN=...` injection — gambles on the
   active account)
- ❌ Forging the author in a commit message

### 2.5 Commit message convention

Historical commits in this repo follow the form `dev(<who>): <msg>`.
Keep this prefix, and derive `<who>` **dynamically** from this repo's
local `user.name` (resolved via §2.4.1) — do not hard-code an identity
from memory or from `git log` of past authors. Per §2.1, the `<msg>`
part must be written in English.
