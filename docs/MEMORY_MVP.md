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
    -> MemoryWriter / optional SceneExtractor
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
histogram in RAM to detect a coarse appearance change. This is not object or
place recognition.

Every new place is an uncertain hypothesis. A repeated place label does not
merge nodes. A forward command followed by a changed RGB histogram can produce
an uncertain `possible_transition` event. A different extracted place label
after such a command can create two distinct place hypotheses and a
`possible_transition` edge. Neither observation verifies a traversable path.
`MemoryWriter` does not create `connected_to`; `SceneGraph` accepts that relation
only with explicit `verified_traversal` provenance and evidence from its caller.
Visual object detections use `visible_from`, while `contains` requires separate
verified containment evidence. Object names alone never merge instances;
reuse needs a stable instance key. Retrieved relation text labels transition
hypotheses separately from verified traversal and visual observations.
`close_episode` clears all records and pending commands; a fresh episode starts
with empty stores.

The repository has no reusable RGB scene or object semantic extractor.
`NullSceneExtractor` is the production default and supplies no semantic labels.
The first RGB frame alone creates no place or object node and may yield an empty
memory context. Subsequent issued commands and coarse RGB appearance changes
can still produce nonsemantic event records with their provenance.
`SceneExtractor` is the adapter protocol; `MockSceneExtractor` accepts scripted
facts for CPU tests and is never selected automatically. Real semantic landmark
claims require a separately validated local RGB extractor. This MVP makes no
paid API calls and reads no keys. An implementation must clear its own state in
the protocol's `reset_episode` and `close_episode` hooks.

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
