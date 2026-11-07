# Episode-local MemoryManager MVP and Hierarchical V2

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
    -> MemoryRetriever (flat) OR L1/L2/L3 retrieval + MemoryFusion (hierarchical)
    -> MemoryContext
    -> inject_memory_context(original VLN sample)
    -> VLNInferenceEngine.generate -> trajectory/RVQ decoding -> Habitat action
```

`build_vln_traj_sample` still selects the checkpoint's original windowed or
SlowFast video segments and output sentence. The adapter adds escaped, bounded
plain text after the original `</navigation_task>` tag when retrieval has a
record. An empty context adds nothing. It does not modify the original
instruction, `<video>` placeholders, special action tokens, or RVQ decoder.
Retrieval shows one representation per detected object and step, while keeping
the original events and graph nodes. Current visual observations receive a
ranking bonus; historical observations remain eligible through instruction
relevance, confidence, importance, and recency. At most one recent action or
turn is shown alongside retrieved observations.
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

## Hierarchical Memory V2

`--memory_strategy flat` is the default and keeps the existing Retriever V2
path. `--memory_strategy hierarchical` runs one `MemoryWriter.observe` per RGB
frame, then indexes that writer's events in three distinct episode-local stores:

- **L1 short:** `WorkingMemoryStore` keeps a bounded deque of step-specific
  detections, their event/node evidence references, recent executed commands,
  and an uncertain place hypothesis when one is supported. A carried sighting
  says when it was last detected; it never claims a detection in the live frame.
- **L2 mid:** `EpisodicMemory` retains every original event and separately
  indexes key events and time segments. Its own retrieval ranks visual and
  navigation events, without letting repeated ordinary actions fill the result.
- **L3 long:** `SemanticIndex` groups independent sightings by **category**,
  recording their first/latest steps, count, confidence statistics, instruction
  relevance, and links back to the original events and graph nodes. A shared
  label is not a physical instance identity. Only existing evidenced graph
  relations can be retrieved; indexing creates no spatial edge.

When `--memory_place_enabled` is set with hierarchical `long`, L3 retrieval
uses `TopologicalMemory` instead. `SemanticIndex` remains available for the
existing hierarchical mode without place recognition, so earlier experiments
retain their original retrieval behavior.

`MemoryManager.retrieve_layers()` queries each enabled store independently.
`MemoryFusion` then presents current observations, historical events, and
category summaries with layer/step/confidence/source/evidence labels. It uses
the original observation IDs and node IDs to avoid repeating one detection
across layers, while preserving observations from different steps. The same
`inject_memory_context` boundary inserts the fused text after the original
navigation instruction. The 600-character text setting is unchanged; no visual
history sampling, token accounting, or action decoding changes are made.

Use `--memory_layers short,mid,long` (default) or any nonempty subset for the
seven layer ablations. For example, `--memory_layers mid,long` runs without L1.
The following CPU Mock command prints independent layer records and fused
results at steps 5 and 20 without loading a model, simulator, or RGB files:

```bash
PYTHONPATH=src python3 scripts/demo_hierarchical_memory.py --layers short,mid,long
```

For one AutoDL R2R episode using the existing local YOLO weights, add
`--memory_strategy hierarchical --memory_layers short,mid,long` to the
`CLIENT_ARGS` in the YOLO command below. `--verbose` includes `layer_contexts`
and the final `injected_memory_text` in each `[memory]` JSON line.

## RGB visual place graph (optional L3)

The optional path is `RGB -> local DINOv2-Small patch tokens -> mean pooling and
L2 normalization -> cosine place match -> episode-local PlaceNode/PlaceEdge ->
L3 retrieval -> MemoryFusion -> existing Prompt adapter`. The model loads on
the first scheduled extraction, never when memory or the long place layer is
off. The directory must already contain Hugging Face `config.json`,
`preprocessor_config.json`, and local model weights; processor and model loading
use `local_files_only=True`. No images are saved. Existing YOLO extraction runs
once through `MemoryWriter`; the descriptor extractor reads the same current RGB
only at steps 0, 5, 10, ... by default.

A `PlaceNode` stores a normalized visual descriptor, first/latest observation
steps, observation and visit counts, YOLO category summaries, and original
event/node references. The category summary keeps each detection box as a
separate observation; matching category names never establish physical instance
identity. A `PlaceEdge` is directed `observed_transition` between two different
recognized visual places. It requires an intervening executed-action event and
records the previous/current descriptor observations plus action evidence.
It **does not** verify a traversable connection or geometric adjacency.
Without such motion evidence, a changed visual descriptor creates no edge.
Matching a prior visual place after an unsupported view change updates its
observation count without asserting a new physical visit.
The default cosine threshold `0.85` is an initial setting awaiting Habitat
calibration; visual aliasing and viewpoint changes can cause false matches or
splits. The original SceneGraph's place hypotheses and spatial relations remain
separate from these DINO visual clusters.

L3 retrieval marks a descriptor from the current step as `current_place`; a
carried visual place from an earlier step is `last_recognized_place`. It can
return observed transition neighbors, evidence-backed historical revisits,
and YOLO categories tied to the same descriptor-observed RGB step. When YOLO
detects a class on a step without a place descriptor, that detection remains
in episodic memory rather than being assigned to an unverified place.
Fusion prioritizes the current visual place and removes repeated YOLO evidence
across L1/L2/L3. The existing 600-character text cap is unchanged.

In an inference environment with NumPy, this scripted Mock demo prints the
step 0/5/10/20 place graph, three layer results, and fused Prompt text without
loading DINO weights or Habitat:

```bash
PYTHONPATH=src python3 scripts/demo_topological_memory.py
```

After placing a real local DINOv2-Small directory and YOLO11n `.pt` file on
AutoDL, use the existing one-episode Habitat setup with these client flags:

```bash
MODEL_PATH=/path/to/LightNav-0 GPU_IDS=0 EPISODES=1 MAX_STEPS=40 \
  CLIENT_ARGS='--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long --memory_extractor yolo11 --memory_extractor_model_path /path/to/local/yolo11n.pt --memory_place_enabled --memory_place_model_path /path/to/local/dinov2-small --memory_place_extract_every_n_steps 5 --memory_place_similarity_threshold 0.85 --memory_place_device cpu --verbose' \
  bash scripts/eval_habitat.sh
```

`[memory]` JSON lines include current `place_id`, top-1 `place_similarity`,
creation/match/revisit/transition status, topological graph counts, place observation
and visit counts, L3 retrieval count, errors, and the actual injected text.
This task has not validated real DINO descriptors or the 0.85 threshold on
Habitat; compare these logs against observed RGB before interpreting loop
closures as real locations.

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
`graph_nodes`, `graph_edges`, `retrieved_records`, `injected_records`, and the actual
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
