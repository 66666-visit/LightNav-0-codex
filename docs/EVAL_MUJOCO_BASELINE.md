# MuJoCo VLN Baseline evaluation

This evaluation runs **outside** the LightNav inference service. The separate
runner drives the MuJoCo demo through its existing browser WebSocket, reads the
simulator pose from runtime snapshots, and relays each existing model WebSocket
request through a local observation proxy. The proxy records the request and
response without changing their contents. The runner writes one JSONL record
per task run plus a summary. It does not change `src/` or the SlowFast selector.

## Before collecting a success rate

The bundled demo has one scene (`procthor-val-2`) and a fixed TurtleBot spawn at
`(x=6.5, y=13.8, yaw=0)`. Its `reset` command restores that spawn. The simulator exposes
the robot's world-frame `pose.x` and `pose.y`, but it has **no built-in navigation
goal or success metric**. Model `stop=true` only means the model requested a stop.

The three positions in
[`scripts/mujoco_baseline_tasks.example.json`](../scripts/mujoco_baseline_tasks.example.json)
are **candidate** points near the spawn, chosen from the scene layout. They have
not been checked on AutoDL for free space, reachability, camera interpretation,
or whether the instruction names the intended destination. Every example task
sets `goal_verified=false`; it must not contribute to a reported success rate.

For a usable fixed task set, inspect the scene and first-person camera, drive
to each intended destination, record the robot-center `(x, y)` from `/api/health`,
and check that the target circle lies in traversable space and is outside the
spawn circle. Adjust `goal_xy` and `radius_m` to the intended stopping region;
then set `goal_verified=true`. Keep this verified JSON file fixed across all
strategies and archive it with the results. A task is successful only when the
model stops, the final robot-center distance to its verified target is at most
`radius_m`, **and** the sampled trajectory passes the collision check. A STOP
outside the region or a detected collision is a failure. The runner also reports
the minimum distance seen in sampled snapshots; entering the region without STOP
is not strict success.

## AutoDL run

Start `lightnav-serve` with the intended checkpoint and default
`--aspect_mode stretch` (or reuse the already running service):

```bash
PORT=8051 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
  --task vln --model_path checkpoints/LightNav-0 --backend vllm_local
```

The service's SlowFast tiers, video processor, video size, and pooling settings
must match the diagnostic JSON; server-side overrides make the derived Token
budget invalid. Start the demo and evaluator in separate terminals:

```bash
cd mujoco_demo && MUJOCO_GL=egl ./run.sh --vln-server ws://127.0.0.1:8051
```

```bash
cd mujoco_demo && uv run --no-sync python ../scripts/eval_mujoco_baseline.py \
  --demo-url http://127.0.0.1:8088 \
  --vln-server ws://127.0.0.1:8051 \
  --tasks /path/to/verified_tasks.json \
  --budget-diagnostic ../outputs/diagnostics/slowfast_budget_20261029.json \
  --model-id LightNav-0-baseline \
  --repeat 3 --timeout-s 180 --max-steps 120 \
  --output ../outputs/eval_mujoco/baseline_run1
```

The example task file is deliberately unverified. Verify its targets and save
the reviewed copy at the path passed to `--tasks` before running an experiment.
The runner resets MuJoCo before each repetition. After the reset acknowledgment,
it waits for two distinct camera timestamps later than the acknowledgment, then
checks spawn position and yaw. One image captured before reset can receive a
later timestamp while it is being encoded, so the first post-ack image is
discarded. The instruction starts after this check.
Run only one evaluator against a demo instance at a time; the demo gives VLN
control to one WebSocket connection. `--model-id` is a label supplied by the
operator for traceability; the runner does not inspect or authenticate the
model weights behind the WebSocket URL.

## Output and metric definitions

The `--output` path must be a new directory: the evaluator creates
`results.jsonl` and `summary.json` inside it and refuses to overwrite an
existing run. `results.jsonl` contains one JSON object per repetition: task ID,
instruction, target and radius, repeat index, reset/final pose, duration, stop reason,
model STOP, strict success, `unscorable_reason`, collision-check result and
pose samples, inference step count, history-frame count, latency, and visual-token
budget. `summary.json` reports success over verified attempts,
navigation steps and duration over completed trials, and latency over completed
model responses. Keep the JSONL file: it preserves failures and per-run details
behind each mean.

- **Success rate** is successful verified attempts divided by all verified
  attempts; runtime failures count as unsuccessful attempts. The summary also
  gives `valid_scored_success_rate`, `unscorable_verified_trials`,
  `scoring_coverage`, `abnormal_trials`, and reason counts. Model STOP alone is
  never a success signal. Unverified tasks do not enter either rate.
- **Trajectory validity** uses a separate copy of the bundled MuJoCo TurtleBot
  model. The checker samples the recorded `(x, y, z, yaw)` path at up to 0.02 m
  intervals, calls `mj_forward`, and looks for penetrating contacts between the
  robot's collidable base, wheels, or caster and non-floor collision geoms. It first
  checks that spawn is clear and a known wall is collidable. Missing or sparse
  pose samples, geometry load failure, or failed self-check yield
  `trajectory_valid=null`; a close STOP then has `success=null`, not a trusted
  success. A detected wall or furniture penetration yields `success=false`.
  This is a dense sampled contact test, not a mathematical continuous-collision
  proof. It supports the bundled TurtleBot; other robot backends remain unscored.
- **Navigation steps** count completed model predictions, not MuJoCo physics
  ticks or MPC control cycles. The run also records its elapsed wall time.
- **Latency** has two distinct measurements. `server_latency_ms` comes from
  LightNav's `next` response and measures the server's prediction path.
  `proxy_roundtrip_ms` times the relay's upstream send/receive and includes
  WebSocket transport and relay overhead. Neither is the demo client's own
  end-to-end timing or pure GPU kernel time. The summary reports separate
  per-step means.
- **History frames** is the number of frames observed by the LightNav session
  at each prediction. The final count and per-step counts describe history
  growth, not the number of frame slots selected by SlowFast. The selected
  slot count can include a duplicated frame used to make a temporal pair.
- **Visual tokens** are derived from the checkpoint's real processor diagnostic
  and the existing `slowfast_segments` selector for the corresponding history
  lengths. Report ViT input patch rows, ViT output tokens before pooling, and
  post-ViT LLM visual tokens separately. These are a processor-calibrated
  **budget**, not live GPU measurements. Post-ViT pooling reduces LLM input
  features; it does not reduce ViT work by the same ratio. The complete input
  sequence length is measured only in the separate diagnostic, not by this
  runtime evaluator. If a prompt exceeds `max_seq_len`, the budget cannot be
  interpreted as the tokens the model actually consumed; check the diagnostic's
  prompt-fit fields for the intended instructions.

The evaluator rejects or marks unavailable metrics when it sees unverified
goals, inconsistent SlowFast calibration, a noncontinuous prediction sequence,
trajectory gaps, geometry failure, or a scene/robot mismatch. It cannot inspect
the weights behind the model URL;
the operator must keep the launch command and checkpoint fixed. Unavailable
success and token counts are not replaced with zero.

## Comparing later sampling strategies

Use the same verified tasks, repeat count, spawn, robot backend, scene, goal
radii, wall-time and step caps, camera, LightNav checkpoint, server backend, and
server options. Keep `--aspect_mode stretch` and do not override processor or
SlowFast configuration while using this diagnostic calibration. Record the
checkpoint identity in `--model-id` and retain the diagnostic JSON used for
token calibration. Run variants serially against the
same demo and GPU setup. Distinguish changes in visual-token budget from
changes in latency: ViT cache hits, WebSocket transport, scheduling, and MPC
tracking can also affect wall time.

This baseline does not provide Habitat metrics such as SPL or geodesic distance.
The target circles are task annotations made for this MuJoCo scene, and success
claims are limited to their verified geometry. The current demo client expects
the model response to include a `pointing` object; use the same checkpoint that
already runs successfully in the interactive MuJoCo demo.

On AutoDL, run the protocol and geometry checks in the MuJoCo environment:

```bash
cd mujoco_demo && uv run --no-sync python -m unittest discover -s ../tests \
  -p 'test_*mujoco_baseline*.py' -v
```
