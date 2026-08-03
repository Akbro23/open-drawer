# open-drawer

Language-conditioned, touch-gated drawer opening. A Franka Panda opens the
drawer a prompt names, pulls it to a stop it cannot see, feels the stop, and
lets go — in Genesis, with pi0.5 conditioned on a fingertip tactile array.

```
"open the left drawer"  ->  approach  ->  descend onto the rail  ->  grip
                        ->  pull  ->  feel the stop  ->  release  ->  retract
```

**Technical report:** [REPORT.md](REPORT.md) — measurements, scaling results,
and the reasoning behind the task design, the tactile pathway and the training
configuration.

**Demonstration video:** _link_

## Overview

A two-drawer cabinet stands free on a table. Three properties make each of the
three modalities load-bearing rather than decorative:

- **Language decides *which*.** The two drawers are geometrically and visually
  identical, so the prompt is the only thing that selects a target.
- **Touch decides *when*.** Each drawer's travel stop is randomized per episode
  and per environment, and nothing about a closed drawer reveals how far it will
  open. It cannot be memorized or seen; it has to be felt.
- **Over-pulling costs something.** The cabinet is not bolted down. Keep pulling
  after the drawer bottoms out and the cabinet slides, which fails the episode —
  so pulling for the maximum duration is not a winning strategy.

The pipeline is three stages: a scripted teacher records demonstrations in
batched simulation, pi0.5 is fine-tuned on them with a tactile pathway added to
its action expert, and the checkpoint is scored closed-loop in the same
simulator.

## Requirements

| | |
|---|---|
| Python | 3.12 exactly (`requires-python = "==3.12.*"`) |
| GPU | ROCm-capable, ≥32 GB for training; collection and evaluation need ~2 GB |
| Disk | ~100 GB — dataset ~256 MB, each training checkpoint ~17 GB |
| Accounts | Hugging Face (with the PaliGemma licence accepted), Weights & Biases |

### Install uv

Everything is driven through [uv](https://docs.astral.sh/uv/), which manages the
Python version, the virtual environment and the dependencies together.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell afterwards, or `source $HOME/.local/bin/env`, so `uv` is on
the path. `uv --version` should answer.

### Dependency specifications

Declared in `pyproject.toml`, resolved in `uv.lock`. Nothing is installed by
hand — `uv sync` reads both.

| Package | Version |
|---|---|
| `torch`, `torchvision`, `torchaudio`, `triton` | ROCm 7.2.1 wheels, pinned by URL to `repo.radeon.com` |
| `genesis-world` | 1.2.3 |
| `lerobot` | with the `training` and `pi` extras |
| `bitsandbytes` | ≥0.50 |

The PyTorch wheels are pinned by exact URL rather than by version specifier, so
the ROCm build is not something the resolver can substitute.

## Setup

### 1. Credentials

```bash
cp .env.example .env
```

Then open `.env` and fill in two values:

- **`HF_TOKEN`** — a *read* token from
  https://huggingface.co/settings/tokens. Used to download `lerobot/pi05_base`
  and its PaliGemma tokenizer.
- **`WANDB_API_KEY`** — from https://wandb.ai/authorize. Training logs only;
  pass `--wandb.enable=false` to train without one.

**Then accept the PaliGemma licence** at
https://huggingface.co/google/paligemma-3b-pt-224, signed in as the same
account. This is a separate step from creating the token, and skipping it is the
most common way setup fails: the tokenizer download returns 403 no matter how
valid the token is.

`.env` is gitignored and is read by the next step.

### 2. Environment

```bash
source scripts/instance-env.sh
```

This does three things:

1. **Points the uv and Hugging Face caches at persistent storage.** A
   container's own filesystem does not survive a restart, and pi0.5 plus its
   tokenizer are several GB that are not worth fetching twice.
2. **Loads `.env`**, exporting `HF_TOKEN` and `WANDB_API_KEY`.
3. **Switches package and model downloads to mirrors** — `UV_DEFAULT_INDEX` to
   the Tsinghua PyPI mirror and `HF_ENDPOINT` to `hf-mirror.com`. This matters
   **if you are running from mainland China**, where PyPI and `huggingface.co`
   are effectively unreachable. Outside China, either skip this step entirely or
   export your own values first: every variable it sets honours one already
   present, so `UV_DEFAULT_INDEX=https://pypi.org/simple` set beforehand wins.

It prints what it set, reporting the two secrets as `set` or `MISSING` without
echoing them, and appends itself to `~/.bashrc` so later shells pick it up.

### 3. Install

```bash
uv sync
```

Installs the ROCm PyTorch stack, Genesis, lerobot and bitsandbytes, and installs
this project itself so its commands (`render`, `rollout`, `collect`, `train`,
`evaluate`) exist on the path.

**Source step 2 before this.** Both mirrors replace their upstream rather than
adding to it, and `UV_DEFAULT_INDEX` decides the URLs written into `uv.lock` —
so a lock produced in a shell that never sourced it points at the wrong index.

### 4. Check it works

```bash
uv run render --envs 1
```

Runs one scripted episode and writes `out/episode.mp4` and
`out/episode_wrist.mp4`, plus a dump of the derived geometry. It should report
`success=True`. This exercises the simulator, the renderer and the tactile
sensors without needing any model weights, so it separates setup problems from
model problems.

## Reproducing the results

Three stages, in order. Each depends on the previous one's output, and the
default paths chain automatically.

### 1. Collect the dataset

```bash
uv run collect
```

Runs the scripted teacher over 1024 episodes — 8 batches of 128 environments —
and writes them in LeRobot format to `data/open_drawer`. Only successful
episodes are recorded. **Takes about 80 minutes** and produces ~256 MB.

Then verify the dataset means what the deployment loop thinks it means:

```bash
uv run evaluate
```

This is the replay regression: it feeds the teacher's own recorded actions back
through the inference loop and checks that the episodes reproduce. It needs no
checkpoint and takes under a minute. **If it fails, do not train** — the
recorded actions and the evaluation loop disagree, and a policy trained on them
would be solving a different control problem.

### 2. Train

```bash
scripts/train.sh
```

Fine-tunes pi0.5 with the tactile pathway for 10000 steps at batch 16, using an
8-bit AdamW. **Takes about 10 hours.** The script launches it detached from the
terminal, so the session can be closed; it prints the log path and the commands
to follow, check and stop the run. Checkpoints land in
`out/train/open_drawer/checkpoints/` every 2500 steps.

```bash
tail -f out/train-<timestamp>.log        # follow
ps -p $(cat out/train.pid)               # still alive?
scripts/train.sh --resume=true           # continue after a crash
```

To run in the foreground, or to change anything, call the command directly —
every lerobot flag is passed through and overrides the defaults:

```bash
uv run train --steps=2000 --batch_size=8 --wandb.enable=false
```

In wandb, `tactile_cond_norm` and `tactile_cond_ratio` report the magnitude of
the tactile contribution to the action expert's conditioning. Both start at
exactly zero by construction, so their climbing is the evidence that touch is
being used at all.

### 3. Evaluate

```bash
uv run evaluate --mode policy --envs 8
```

Runs the trained checkpoint closed-loop in the simulator and scores it by the
same criteria as the teacher: the target drawer reached its stop, the gripper
released, the other drawer never moved, and the cabinet stayed put. It also
reports **release latency** — control steps between the true stop and the
release — which is the measure of whether the tactile channel is doing its job.

It reads `out/train/open_drawer/checkpoints/last/pretrained_model` by default;
pass `--checkpoint` for any other. Add `--video out/eval.mp4` to film one
environment of the first batch, with the live task state burned in — this is
how the demonstration video above was made.

> Do not evaluate while training is running. Both load a 4B model onto the same
> GPU.

## Command reference

| Command | What it does |
|---|---|
| `uv run render` | Film one teacher episode to mp4, with live task state burned in |
| `uv run rollout` | Teacher success rate and physics throughput; `--scaling 32,64,128` sweeps |
| `uv run collect` | Record the LeRobot dataset; `--scaling 8,16,32` probes the host-RAM ceiling |
| `uv run train` | Fine-tune pi0.5 + tactile |
| `uv run evaluate` | Replay regression, or `--mode policy` to score a checkpoint |
| `scripts/instance-env.sh` | Caches, secrets and mirrors (source it) |
| `scripts/train.sh` | Launch training detached, with disk and duplicate-run checks |

Every command takes `--help`.

## Repository map

```
config.py          every tunable, frozen dataclasses; asserts its own geometry
assets.py          generated cabinet MJCF: carcass, two drawers, rails
randomize.py       target side, per-drawer stops, cabinet pose, grasp residual
scene.py           batched build and reset; per-env travel limits
robot.py           batched IK, velocity-limited moves, the control tick
tactile.py         taxel grids, the 24-dim feature, the peak reading
task_state.py      rail pose, success latch, release latency
teacher.py         the six phases, in lockstep across environments
record.py          (observation, action) pairs at 25 Hz
collect.py         LeRobot dataset writer, plus a RAM scaling probe
policy_tactile.py  pi0.5 with tactile wired into the action expert
train.py           lerobot-train wrapper with an 8-bit optimizer
eval.py            policy in the loop; the replay regression
rollout.py         success rate and throughput
render_episode.py  mp4 plus the derived-geometry dump
```

Generated directories — `assets/` (rewritten from config on every scene build),
`data/` and `out/` — are gitignored.
