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
    ├── onnx.py                ONNX export (YOLO/RTDETR), with simplify
    └── utils/
        ├── __init__.py        re-exports the utilities below
        ├── cli.py             GLOBAL_CONTEXT_SETTINGS, print_version
        ├── execute.py         delayed_execution (threading.Timer for late metadata writes)
        ├── f1plot.py          OCR the best threshold off the F1-curve image
        ├── md.py              markdown table -> DataFrame
        └── pe.py              pretty-print numbers as k/M/G
```

### 1.2 Core data flow

1. **Train**: `yolov8.train.train_object_detection` / `train_segmentation`.
   - Picks the model family (v8/9/10/11/12 use `YOLO`, `rtdetr` uses
     `RTDETR`), conventionally writes to `runs/<task_name>/`, and uses
     `delayed_execution` to write `model_type.json` **30 seconds later**.
     This is because Ultralytics wipes the workdir at training start, so
     the metadata file must be written *after* that wipe. Do not rewrite
     this as a synchronous pre-train write.
   - If `weights/last.pt` exists, training auto-resumes (`resume=True`).
2. **Export**: `yolov8.export.export_model_from_workdir(workdir, export_dir, ...)`
   - When copying `weights/best.pt`, **`train_args.data` / `project` / `model`
     are sha3-anonymised** to avoid leaking the trainer's local paths into
     published artifacts. Do not revert this to plaintext.
   - `yolov8.utils.f1plot.get_f1_and_threshold_from_image` OCRs the best
     threshold off the F1 curve image and writes `threshold.json`.
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
- Do **not** turn the
  `delayed_execution(_writing_model_type_file, delay_seconds=30)` call in
  `yolov8/train/object_detection.py` and `yolov8/train/segmentation.py`
  into a synchronous write — Ultralytics wipes the workdir at training
  start, so the late write is load-bearing.
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
