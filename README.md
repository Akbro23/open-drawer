# open-drawer

Language-conditioned, touch-gated drawer opening. A Franka Panda opens the
drawer a prompt names, pulls it to a stop it cannot see, feels the stop, and
lets go — in Genesis, with pi0.5 conditioned on a fingertip tactile array.

```
"open the left drawer"  ->  approach  ->  descend onto the rail  ->  grip
                        ->  pull  ->  feel the stop  ->  release  ->  retract
```

## The task, and why it is not trivially solvable

A two-drawer cabinet stands **free** on a table. Three properties make each of
the three modalities load-bearing rather than decorative:

**Language decides *which*.** The two drawers are geometrically and visually
identical. No camera can tell them apart, so the prompt is the only thing that
selects a target. A policy that ignores it scores 50%.

**Touch decides *when*.** Each drawer's travel stop is randomized per episode
and written per environment. It is *invisible* — nothing about a closed drawer
reveals how far it will open — so how far to pull cannot be memorized or seen.
It has to be felt.

**Over-pulling costs something.** This is the part that makes the previous
point real. The cabinet is not bolted down. Keep pulling after the drawer has
bottomed out and the load transfers into the cabinet, which slides toward the
robot. Success requires it to have stayed put. Without this, a policy that
simply pulls for the maximum duration and then opens its jaws would score 100%
and never touch the tactile channel.

## Setup

Requires Python 3.12 and a ROCm-capable GPU.

```bash
cp .env.example .env          # fill in HF_TOKEN and WANDB_API_KEY
source scripts/instance-env.sh
uv sync
```

`scripts/instance-env.sh` does three things: points the uv and Hugging Face
caches at persistent storage (the container's own filesystem does not survive a
restart), loads `.env`, and — **if you are running from mainland China** —
switches package and model downloads to mirrors, since PyPI and
`huggingface.co` are effectively unreachable there. It sets
`UV_DEFAULT_INDEX` to the Tsinghua PyPI mirror and `HF_ENDPOINT` to
`hf-mirror.com`.

Both replace their upstream rather than adding to it, so **source it before
`uv sync`**: `UV_DEFAULT_INDEX` decides the URLs written into `uv.lock`. The
script adds itself to `~/.bashrc` so later shells pick it up.

Outside China, skip it or export your own values first — every variable it sets
honours one already in the environment, so `UV_DEFAULT_INDEX=https://pypi.org/simple`
set beforehand wins.

`uv sync` then installs the pinned ROCm PyTorch stack, Genesis 1.2.3 and
lerobot, and installs this project so its commands exist.

pi0.5 additionally needs the PaliGemma licence accepted in a browser, not just
a token — otherwise its tokenizer returns 403 regardless of what the token
says.

## Pipeline

```bash
uv run render                          # film one episode, dump derived geometry
uv run rollout --envs 16               # teacher success rate and throughput
uv run evaluate                        # replay regression (no checkpoint needed)

uv run collect --scaling 8,16,32,64    # find the host-RAM ceiling first
uv run collect --envs 128 --batches 8  # record the dataset
uv run train                           # fine-tune pi0.5 + tactile
uv run evaluate --mode policy          # score the checkpoint in the loop
```

`uv run rollout --scaling 1,4,16,64` sweeps batched physics with rendering off.

## Results

Teacher, 128 episodes at N=64:

| | |
|---|---|
| success | 128/128 |
| opened / released | 128/128 |
| wrong drawer, cabinet dragged | 0, 0 |
| travel | mean 95.9 mm |
| shortfall from the stop | mean 0.00 mm |
| cabinet displacement | mean 0.25 mm, max 0.68 mm (limit 10 mm) |
| release latency | median 132 control steps, max 205 |

Batched physics, two episodes per env count — 960 episodes, all successful:

| N | wall | episodes/min | speedup |
|---|---|---|---|
| 32 | 31.6 s | 121.5 | 1.0× |
| 64 | 31.3 s | 245.6 | 2.0× |
| 128 | 33.3 s | 461.3 | 3.8× |
| 256 | 34.8 s | 882.7 | 7.3× |

Eight times the environments for 10% more wall time. Collection is bounded by
host RAM rather than by this, which is why `collect` has its own probe.

Dataset: **1024 episodes, 170,624 frames**, 0 dropped, in 81.7 min. Only ~5 of
those minutes are simulation — the rest is the LeRobot writer encoding video at
~27 ms/frame, which is what actually bounds collection.

Replay regression, 16 episodes — the teacher's own recorded actions fed back
through the inference loop:

| | teacher | replay |
|---|---|---|
| success | 16/16 | 16/16 |
| mean travel | 94.6 mm | 94.6 mm |

Max travel difference **0.00 mm**. This is the cheapest test in the project and
the one that catches the most: if replay diverged, the recorded actions would
not mean what the eval loop thinks they mean, and the dataset and the
deployment would be different control problems.

## How it works

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

### Three contracts

**25 Hz is the control rate, not a storage setting.** `robot.apply` issues one
command per tick and holds it, so the teacher and the policy solve the same
control problem. Changing it invalidates the dataset.

**An action is a command, not a motion.** The recorded value is
`q_cmd - q_measured`, which carries the standing error a position-controlled
arm needs merely to hold itself up against gravity. Recording achieved motion
would label every hold "do nothing", and a policy obeying that at inference
releases the offset holding the arm up and sags further every step.

**Tactile is its own dataset feature.** It is never folded into
`observation.state`. pi0.5 does not feed state to the action expert as a
vector — it normalizes it, digitizes it into 256 bins and pastes it into the
text prompt. A contact signal sent through that path arrives 8-bit quantized,
and one that is quiet for most of an episode and then steps would collapse into
a bin or two while the event saturates. Kept separate it stays continuous.

### Where touch enters the policy

`policy_tactile.py` subclasses pi0.5 and adds a small MLP whose output is added
to `adarms_cond` — the single vector that modulates every adaRMSNorm layer in
the action expert. That puts touch on a continuous path straight into the
action decoder with no tokenizer in between. The encoder's output layer is
**zero-initialised**, so on step one the model is bit-identical to pretrained
pi0.5 and the tactile pathway grows from nothing rather than injecting noise
into a model that already works.

## Design notes worth knowing

**KinematicTaxel measures penetration, not force.** Its reading is a spring on
how deep the virtual probes sit inside what they touch. Against a rigid rail
the pads sink by micrometres under load, and the reading is *flat* — measured
at 0.19 N through an entire pull, unchanged when the drawer jammed. Two changes
made the sensor work: giving the rail a compliant contact so the pads sink
measurably under load, and cutting the grip squeeze so that load dominates the
reading instead of a large constant preload. The bands are now 0.16–0.23 while
sliding and 0.28–0.46 against the stop, stepping at each environment's own
randomized stop. The release threshold sits in that gap and is **measured, not
chosen**.

**The unanchored cabinet is a force limiter.** Everything the arm pulls passes
through it, so its static friction `mu * m * g` caps the force anywhere in the
chain — including at the fingertip, and therefore the top of the sensor's
range. That puts the design inside a window: the ceiling must sit well above
the drawer's own sliding friction, or the cabinet creeps during a legitimate
pull, and well below what the arm can deliver, or it never moves and
over-pulling costs nothing. Both bounds were hit while tuning.

**Grip force is `kp` times commanded overshoot.** The jaws are commanded
*through* the rail; they cannot get there, and the standing error is the grip.
Commanding the rail's own thickness would arrive with no error and apply no
force at all.

**The wrist camera mount does not transfer from a tool-down task.** The grasp
orientation carries 90 degrees of wrist yaw, which maps the hand's y axis onto
world x. A tilt applied as pitch therefore pans the view sideways and never
toward the cabinet; the lean has to be roll. A 180 degree roll additionally
inverts the column that `T_to_pos_lookat_up` reads as "up", rendering the scene
upside down. Both were found by rendering, not by reasoning.

## Status

Exercised: scene construction, per-environment travel stops, the teacher, the
recorder's shapes against the declared dataset features, the LeRobot write path
(1024 episodes, none dropped), the replay regression against the dataset on
disk, and training — `policy_tactile.py` loads `lerobot/pi05_base` and steps
under the 8-bit optimizer.

`observation.tactile` lands where it was meant to. `dataset_to_policy_features`
types any `observation.*` key as STATE, so it becomes a policy input and picks
up normalization statistics, while the state tokenizer reads `observation.state`
by name and never sees it. Normalized, and not quantized into the prompt.

That confirms the plumbing, not the mechanism. The tactile encoder is
zero-initialised, so a training run looks identical whether or not touch is
carrying information — only closed-loop evaluation, and release latency in
particular, can show that it is.

Not yet done: the full training run, and evaluation of its checkpoint.
Inference latency per chunk is unmeasured.

The 100% teacher rate is over roughly 1100 episodes.
