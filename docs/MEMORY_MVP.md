# Episode-local MemoryManager MVP

## Scope and call chain

The memory switch belongs to the Habitat R2R evaluation client:
`HabitatEvalConfig.memory.enabled` (CLI `--memory_enabled`, default `false`).
`run_habitat_eval` resets `TrajVocabVLNCEPolicy` for every accepted episode and
closes its memory after the episode, including on errors. The policy calls
`MemoryManager.observe` with the current HWC `uint8` RGB frame, original
instruction, and its own previous command/prediction. Habitat `info` is not
passed to memory; goal distance, pose, success, and oracle data remain in the
evaluator for metrics only.

```text
Habitat RGB + instruction + previous policy command/prediction
    -> ObservationPacket (RGB is transient)
    -> MemoryWriter / NullSceneExtractor or local YOLOSceneExtractor
    -> WorkingMemoryStore + EpisodicMemory + SceneGraph
    -> MemoryRetriever (instruction, place, recency, quality; top_k/char cap)
    -> MemoryContext
    -> inject_memory_context(original VLN sample)
    -> VLNInferenceEngine.generate -> trajectory/RVQ decoding -> Habitat action
```

`build_vln_traj_sample` still selects the checkpoint's original windowed or
SlowFast video segments and output sentence. The adapter adds escaped, bounded
plain text after the original `</navigation_task>` tag when retrieval has a
record. An empty context adds nothing. It does not modify the original
instruction, `<video>` placeholders, special action tokens, or RVQ decoder.
The limit is 600 characters by default; it is not a tokenizer or multimodal
context-length guarantee. Real context-length behavior remains to be checked
during AutoDL inference. With the switch off, `policy.act` uses its original
`generate_from_frames` call.

## Records and evidence

`schemas.py` defines JSON-compatible working state, events, graph nodes and
edges, queries, contexts, and an observation packet. Packet serialization
deliberately excludes RGB. Events and graph records have source and confidence.
No frame is saved by this module; the writer retains only a tiny RGB color
histogram in RAM to detect a coarse appearance change. The optional YOLO11n
extractor processes the transient HWC `uint8` RGB frame and reports only object
class detections with per-box confidence and optional box coordinates.

Every new place is an uncertain hypothesis. A repeated place label does not
merge nodes. A forward command followed by a changed RGB histogram can produce
an uncertain `possible_transition` event. A different extracted place label
after such a command can create two distinct place hypotheses and a
`possible_transition` edge. Neither observation verifies a traversable path.
`MemoryWriter` does not create `connected_to`; `SceneGraph` accepts that relation
only with explicit `verified_traversal` provenance and evidence from its caller.
Visual object detections with an evidenced place use `visible_from`; detections
without place evidence become unanchored object nodes and create no relation.
`contains` requires separate verified containment evidence. Object names alone never merge instances;
reuse needs a stable instance key. Retrieved relation text labels transition
hypotheses separately from verified traversal and visual observations.
`close_episode` clears all records and pending commands; a fresh episode starts
with empty stores.

`NullSceneExtractor` remains the default and supplies no semantic labels.
The first RGB frame alone creates no place or object node and may yield an empty
memory context. Subsequent issued commands and coarse RGB appearance changes
can still produce nonsemantic event records with their provenance.
`SceneExtractor` is the adapter protocol; `MockSceneExtractor` accepts scripted
facts for CPU tests and is never selected automatically. `YOLOSceneExtractor`
is selected only with `--memory_enabled --memory_extractor yolo11` and an existing
local weights file. It converts RGB to the BGR array expected by the Ultralytics
NumPy API, runs on steps 0, 5, 10, ... by default, and returns an empty semantic
result for skipped steps or failed/empty detection. It never produces
`place_label`, `contains`, or `connected_to`. COCO-style object detection does
not establish room type, object identity across frames, or navigation topology.
The module makes no paid API calls and reads no keys.

## Local YOLO11n setup and diagnostics

In the existing Python 3.11 inference environment on AutoDL, install the
optional dependency and provide an already obtained local pretrained weight
file. The project does not download weights automatically; the extractor checks
that the local `.pt` file exists and loads as a detection model before inference.

```bash
cd /path/to/LightNav-0-codex
python -m pip install -e '.[yolo11]'
mkdir -p /path/to/local
cp /path/to/already-downloaded/yolo11n.pt /path/to/local/yolo11n.pt
test -s /path/to/local/yolo11n.pt
```

Run a single R2R episode using the same checkpoint and Habitat setup as the
baseline. Set `CLIENT_PYTHON` if the inference interpreter is not the script's
default `.venv/bin/python`. This first check keeps YOLO on CPU to avoid competing
with the navigation model and Habitat for GPU memory:

```bash
MODEL_PATH=/path/to/LightNav-0 GPU_IDS=0 EPISODES=1 MAX_STEPS=40 \
  CLIENT_ARGS='--memory_enabled --memory_extractor yolo11 --memory_extractor_model_path /path/to/local/yolo11n.pt --memory_extractor_device cpu --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5 --verbose' \
  bash scripts/eval_habitat.sh
```

The printed output directory contains the client log under `logs/` and episode
metrics under `shard_0/results.jsonl`. Each `[memory]` JSON line reports `step`,
`extraction_attempted`, detected `objects` and confidence, `events_written`,
`graph_nodes`, `graph_edges`, `retrieved_records`, and the actual
`injected_memory_text` with `injected_memory_chars`. No raw RGB is written by
MemoryManager. The local weights must really be YOLO11n; the adapter cannot
verify a renamed weight file's architecture. This command has not been run with
real weights or Habitat in the CPU development environment.

## Test and smoke commands

CPU tests in a Python 3.11 inference environment:

```bash
pip install -e '.[test,habitat]'
make check
make test
```

On AutoDL, after the R2R dataset, Habitat Python 3.9 environment, inference
environment, and checkpoint are already in place, run one episode:

```bash
MODEL_PATH=/path/to/LightNav-0 GPU_IDS=0 EPISODES=1 \
  CLIENT_ARGS='--memory_enabled --memory_top_k 5 --memory_max_context_chars 600 --verbose' \
  bash scripts/eval_habitat.sh
```

Inspect the client log and `shard_0/results.jsonl` under the printed output
directory. Compare with a baseline run on the same episode and seed using the
same command without `--memory_enabled`. The checkpoint was not trained with
memory text, so a prompt-level gain is unverified until measured on real R2R.
