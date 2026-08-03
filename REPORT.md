# Technical Report — open-drawer

**Track 3: Physical AI Challenge**

A Franka Panda opens the drawer a language prompt names, pulls it to a travel
stop it has no way of seeing, detects that stop through fingertip tactile
sensing, and releases — implemented in Genesis and learned with a pi0.5
vision-language-action policy extended with a tactile conditioning pathway.

> Sections marked _Pending_ are measurements that require the full instance
> run; they are left explicit rather than estimated.

---

## 1. Target application

**Articulated-object manipulation with an unknown mechanical limit.**

Opening a drawer is the canonical instance of a much broader industrial
problem: manipulating an object whose kinematics are constrained by a mechanism
you cannot observe. The robot can see a handle; it cannot see the slide rail,
the stop, or how far the mechanism will travel before it binds. The same
structure appears in opening doors, cabinets and panels, extracting trays from
racks, operating levers and valves, and any assembly step that ends when a part
seats rather than after a fixed distance.

The failure mode is what makes it worth solving. A robot that treats "open the
drawer" as a fixed displacement either stops short — leaving the task
incomplete — or keeps pulling into a mechanism that has already bottomed out,
which in the real world means damaged furniture, a torn-out drawer, a dragged
cabinet, or a protective stop on the arm. Neither is acceptable in a home, a
warehouse aisle, or a laboratory.

Our formulation makes all three modalities strictly necessary:

| Modality | Decides | Why it cannot be skipped |
|---|---|---|
| Language | *which* drawer | The two drawers are visually identical; nothing else disambiguates the goal |
| Vision | *where* the handle is | The cabinet's pose is randomized per episode |
| Touch | *when* to stop | The travel stop is randomized and invisible to any camera |

## 2. System architecture and solution design

```
                    Genesis (GPU, batched over N environments)
  ┌────────────────────────────────────────────────────────────────┐
  │  cabinet + Franka Panda + fingertip taxel arrays               │
  │  per-env randomization: target side, travel stops, pose        │
  └───────────────┬──────────────────────────────┬─────────────────┘
                  │                              │
      scripted teacher (privileged)       policy in the loop
                  │                              ▲
                  ▼                              │
         LeRobot dataset  ──────►  pi0.5 + tactile fine-tune
      (wrist RGB, state, tactile,        (bf16, 8-bit AdamW)
       action, language prompt)
```

**Scene.** A two-drawer cabinet stands free on a table, generated
parametrically as a single MJCF body: carcass, two drawers on slide joints, and
a bar rail on each drawer front standing 40 mm proud with a 28 mm slot behind
it. The Panda approaches tool-down, straddles the rail — one finger in front,
one in the slot — closes, and pulls. That geometry is deliberate: it puts the
pull load on the pads as a **normal** force rather than relying on friction.

**Randomization,** per environment, per episode: which drawer the prompt names;
each drawer's travel stop, sampled from 60–130 mm; cabinet position (±20 mm)
and yaw (±8°); and a deliberate 0.5–3 mm misalignment left in by the descent so
the dataset contains corrections rather than only perfect approaches.

**Teacher.** Six phases run in lockstep across all environments — approach,
descend, grip, pull, release, retract — with per-environment branching handled
by masked commands rather than ragged control flow. The teacher is allowed
privileged simulator state, but deliberately does **not** use it for the
release: it triggers on the *measured* taxel reading, because a teacher that
released on privileged knowledge would pair identical observations with
different correct actions and make the behaviour unlearnable.

**Scoring.** An episode succeeds only if the target drawer reached its stop, the
gripper released, the other drawer never moved, and the cabinet stayed put —
all held for 8 consecutive control steps. Failure modes are counted separately
(wrong drawer, released short, never released, cabinet dragged), and *release
latency* — control steps between the true stop and the release — is reported as
the measure of how well the tactile channel is doing its job.

## 3. Datasets

**Self-generated.** No external dataset is used. Demonstrations are produced by
the scripted teacher in simulation and written in LeRobot format.

| Field | Shape | Notes |
|---|---|---|
| `observation.images.wrist` | 224×224×3 | single wrist camera, mp4-encoded |
| `observation.state` | 9 | 7 arm joints + 2 finger positions |
| `observation.tactile` | 24 | 2 fingers × 4 taxels × 3 force components |
| `action` | 9 | per-joint command deltas |
| `task` | text | `"open the left drawer"` / `"open the right drawer"` |

Recorded at **25 Hz** (every 4 control steps of a 100 Hz simulation). Only
successful episodes are written.

As collected: **1024 episodes, 170,624 frames**, 166 frames per episode on
average (6.6 s), recorded in 8 batches of 128 environments. No episode was
dropped — the teacher succeeded in all 1024. Generation took **81.7 minutes**
end to end, 12.5 episodes/min; §4 breaks that time down.

Two properties of this dataset are contracts rather than conventions, and both
are documented at the point of use:

- **An action is a command, not an achieved motion** — `q_cmd - q_measured`.
  This preserves the standing offset a position-controlled arm needs simply to
  hold itself against gravity. Recording achieved motion would label every hold
  "do nothing", and a policy obeying that at inference would sag further every
  step.
- **Tactile is a separate feature, never merged into `observation.state`** —
  see §5.

_Pending: final dataset size and on-disk footprint from the collection run._

## 4. Use of AMD Radeon GPU and ROCm

Every stage of the pipeline runs on a single Radeon GPU through ROCm.

**Simulation.** Genesis executes the rigid-body solver, collision detection and
constraint solving on the GPU, batched across N environments in lockstep. The
batching is the point: all randomization is per-environment state, and every
primitive — IK, control, sensing, scoring — is written as a batched array
operation so that N=1 and N=256 take the identical code path. A Python loop
anywhere in the reset or control path would cap throughput regardless of GPU
capability, so there is none.

**Rendering.** Wrist-camera observations are rendered on-GPU per environment
(`env_separate_rigid`), producing one image per environment per control tick
during collection.

**Tactile sensing.** The taxel arrays are evaluated by Genesis's own compiled
GPU kernels, batched over environments alongside the physics — the sensing is
not a CPU-side post-process.

**Training.** pi0.5 (≈4B parameters) is fine-tuned on a single GPU. Three
choices make that fit: bfloat16 weights, gradient checkpointing, and a
bitsandbytes 8-bit AdamW that halves optimizer moment state. The 8-bit
optimizer required a serialization shim, since bitsandbytes' state violates
safetensors' constraints in two ways (a Python-int `step`, and quantization
maps shared by reference across parameters).

**Inference.** Closed-loop evaluation runs policy inference and physics on the
same GPU, with the policy serving action chunks from an internal queue so a
forward pass runs once per chunk rather than once per control step.

Measured batched-physics throughput, two episodes per environment count:

| N | wall | episodes/min | speedup | success |
|---|---|---|---|---|
| 32 | 31.6 s | 121.5 | 1.0× | 64/64 |
| 64 | 31.3 s | 245.6 | 2.0× | 128/128 |
| 128 | 33.3 s | 461.3 | 3.8× | 256/256 |
| 256 | 34.8 s | 882.7 | 7.3× | 512/512 |

Eight times the environments for 10% more wall time. The wall clock is the fixed
step count of one lockstep episode almost independently of N, which is the
result the batched-array design was aiming for; the residual 10% is the only
part of the pipeline that grows with the batch.

Collection is a different curve, and is measured separately for that reason.
Repeating the sweep with the wrist camera on and a recorder attached — the
configuration collection actually uses — separates the two costs, per batch of N
episodes:

| N | physics only | + rendering | render cost | buffered | device |
|---|---|---|---|---|---|
| 32 | 15.8 s | 22.0 s | +6.2 s | 0.8 G | 1.3 G |
| 64 | 15.7 s | 27.8 s | +12.2 s | 1.6 G | 1.5 G |
| 128 | 16.7 s | 38.3 s | +21.7 s | 3.3 G | 1.5 G |

Physics is flat in N; rendering doubles with it. Past N≈64 rendering is the
majority of the cost, so the physics curve above predicts nothing about
collection throughput. Device memory is flat at 1.5 G and never binds — the
constraint is host RAM, which grows at 26 MB per environment because a whole
batch of frames is held until the episode is written.

**What actually dominates dataset generation is neither.** The full run —
1024 episodes in 8 batches of 128 — took 81.7 minutes. Simulation and rendering
account for about 5 of those (38.3 s per batch, measured above). The remaining
**94% is the LeRobot writer**, encoding 170,624 frames of wrist video at roughly
27 ms per frame on CPU.

We keep this rather than presenting the scaling tables alone, because the
tables are accurate about what they measure and misleading about what matters.
Both curves, and the host-RAM ceiling that bounds the second, turn out to
govern 6% of the wall clock: the choice between N=128 and N=256 changes total
collection time by a few percent, not by the 30% the throughput figures imply.
Dataset generation here is a single-threaded video-encoding problem wearing a
GPU-throughput costume, and the only optimization that would move it is
parallel or hardware-accelerated encoding in the writer.

_Pending, from the instance run: training
step time and peak memory; inference latency per chunk against the 160 ms budget
the 16/4 horizon allows._

## 5. Innovations and key technical contributions

**(a) A tactile pathway into pi0.5's action expert.**
pi0.5 does not feed `observation.state` to the action expert as a vector: it
normalizes it, digitizes it into 256 bins, and pastes it into the *text prompt*
alongside the task string. Appending a force signal there would send it through
the tokenizer 8-bit quantized — and our signal is quiet for most of an episode
and then steps, so per-dimension normalization would collapse the quiet
majority into one or two bins while the contact event saturates. Precisely the
wrong channel for the one transient the task turns on.

Instead, a small MLP encodes the 24-dim taxel field and its output is **added to
`adarms_cond`**, the single vector that modulates every adaRMSNorm layer in the
action expert. This puts touch on a continuous path directly into the action
decoder with no tokenizer in between. The encoder's output layer is
**zero-initialised**, so at step 0 the model is bit-identical to pretrained
pi0.5 and the tactile pathway grows from zero rather than perturbing a working
4B model.

**(b) An invisible, per-environment task parameter.**
Each drawer's travel stop is randomized *per environment* using Genesis's
batched DOF-limit facility. This makes "how far to pull" genuinely
unobservable — not merely hard to see — which is what forces the policy to
close the loop on touch rather than on memorized geometry.

**(c) Task design that makes the modality necessary.**
An earlier formulation of this task is degenerate: with a hard stop and no
penalty, a policy that pulls for the maximum duration and then opens its jaws
succeeds every time without ever using touch. We resolved this physically
rather than with a scoring rule — the cabinet is unanchored, so over-pulling
transfers load into it and drags it, and success requires it to have stayed
put. The consequence is visible in the demonstration video, not just in a
metric.

**(d) Empirical characterization of a simulated tactile sensor.**
Genesis's `KinematicTaxel` measures probe *penetration*, not contact force.
Against a rigid handle the pads deform by micrometres under load and the
reading is flat — measured at 0.19 N through an entire pull, unchanged when the
drawer bottomed out. Two changes made it work: a compliant contact on the rail,
so the pads sink measurably under load, and a reduced grip squeeze, so that
load dominates the reading instead of a large constant preload. The bands
became 0.16–0.23 while sliding and 0.28–0.46 against the stop, stepping at each
environment's own stop. **The release threshold is measured from those bands,
not chosen.** We report this as a finding because the naive configuration
produces a sensor that appears to work and silently carries no information.

**(e) A dataset-contract regression.**
Feeding the teacher's own recorded actions back through the inference loop must
reproduce the demonstration exactly. This runs without a checkpoint and
verifies that the recorded actions mean what the deployment loop believes they
mean — a failure mode that otherwise surfaces only after a full training run.
Ours reproduces at 16/16 with 0.00 mm divergence.

**(f) Upstream contributions.**
Two gaps identified while building, both suitable for upstream patches:
Genesis's `RigidEntity` exposes `get_dofs_limit` and every sibling DOF setter
but no `set_dofs_limit`, though the batched solver-level implementation exists;
and LeRobot's policy factory resolves policy classes through a hardcoded
if/elif chain with no registry, so an externally-defined policy cannot be
selected by `--policy.type` without patching.

## 6. Deliverables

- **Source repository** — parametric Genesis environment, scripted teacher,
  dataset collection, policy extension, training and closed-loop evaluation.
- **Reproducibility README** — setup, dependencies, and the command sequence
  that reproduces the reported numbers.
- **Demonstration video** — produced by `uv run render`, with live task state
  (travel against the episode's stop, tactile reading, cabinet displacement)
  burned into the frame.
- **Dataset** — LeRobot-format demonstrations. _Pending._
- **Fine-tuned checkpoint** and closed-loop evaluation results. _Pending._
- **This report.**

## 7. Results to date

Teacher, 128 episodes at N=64:

| | |
|---|---|
| success | 128/128 |
| opened / released | 128/128 / 128/128 |
| wrong drawer / cabinet dragged | 0 / 0 |
| travel | mean 95.9 mm |
| shortfall from each episode's stop | mean 0.00 mm, max 0.00 mm |
| cabinet displacement | mean 0.25 mm, max 0.68 mm (limit 10 mm) |
| release latency | mean 131, median 132, max 205 control steps |

A further 960 episodes across the throughput sweep succeeded without exception.

Release latency is the figure that distinguishes tactile stop detection from a
task that merely completes: it is bounded well inside the 400-step pull budget,
so the release is triggered by the measured contact and not by the pull phase
running out. An episode that never felt its stop would still score as a success
on every other row of this table.

Replay regression: teacher 16/16, replay 16/16, max travel divergence 0.00 mm.

_Pending: trained-policy success rate, and its breakdown by failure mode._

## 8. What we would highlight

**The design is measurement-driven, and the negative results are kept.** The
central mechanism of this project — tactile stop detection — did not work in
its first implementation, and the report says so and explains why. Constants
that matter are traceable to a measurement rather than a guess: the release
threshold comes from measured force bands; the gripper's pad offset is derived
from the robot model rather than assumed; the home pose was solved and then
verified by rendering that both drawers are actually in frame; the cabinet's
mass sits inside a window bounded on one side by creep during legitimate pulls
and on the other by the arm's own wrist-torque limits.

**The task resists shortcuts by construction.** Each modality was checked for a
degenerate solution that bypasses it, and where one existed the *environment*
was changed rather than the scoring.

## 9. Team

_To be completed: team members and their respective contributions._
