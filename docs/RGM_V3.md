# RGM V3: evidence driven navigation memory

RGM V3 is an independent backend for the frozen LightNav policy. The legacy
`--rgm_enabled` backend, Pilot manifests and comparison reader retain their
existing meaning. V3 has protocol version `rgm_v3_v1` and writes a
`rgm_v3_protocol.json` marker in each result directory.

## Scope and identity

`--rgm_v3_scope episode_local` is the default for the first comparison with the
original LightNav input. It clears the online evidence bank, Episode bank and
spatial graph after each Episode. `scene_persistent` retains only observations
made by this process within the same Scene. It is a separate experiment setting;
its results must be reported separately. The bank is in memory and does not
merge observations across evaluation workers or persist to disk.

Every evidence record has a composite identity of `scene_id`, `episode_id`,
`rollout_idx` and local `evidence_id`. A local `event_0` cannot collide with the
same ID in another Episode. Retrieval checks scene and provenance before
considering a record.

During an active Episode, Mid Memory exposes a read-only earlier-step segment
once it contains both an observation and a completed action. The current step
is excluded and its outcome is `unknown`; the segment is labeled as ongoing.
The V3 runner uses the raw Episode ID because the composite evidence key
already contains the Scene ID. Legacy A/C/D Episode IDs are unchanged.

## Data flow

```text
ObservationPacket (instruction, RGB, prior issued command)
  -> MemoryWriter raw events
  -> V3MemoryStore (bounded Short cache, Episode journal, Scene graph)
  -> V3Retriever (visual, semantic, spatial, episodic channels; RRF)
  -> V3PromptFormatter (ordered whole lines within record/character budgets)
  -> Context Builder (inserts after </navigation_task>)
  -> Policy sample.conversations[human].value
  -> Engine.generate -> DataProcessor -> Qwen Processor
  -> input_ids -> HF generate or vLLM prompt_token_ids
```

V3 does not call the legacy Write, Retrieval or Prompt Gates. It stores valid
low confidence detections with their original detector confidence and source.
Instruction overlap affects retrieval rank; it does not erase evidence. Visual
place matching uses the existing DINO descriptor. Spatial links remain scene
scoped. The formatter labels observed, inferred, unknown and traversed records
and preserves evidence references. Current visual observations win exact RRF
ties with history; generic navigation verbs alone do not recall an unrelated
Episode. Historical recency uses write order, since step numbers restart at
every Episode. Historical experience text includes its most recent recorded action
and outcome. It never rewrites the original navigation
instruction. A memory item that cannot fit is omitted as a whole.

An issued action or RGB appearance change does not verify movement.
`observed_transition` and `inferred_relation` retain UNKNOWN traversability.
`verified_traversal` requires independent measured displacement from a trusted
agent sensor, observed between the two endpoint observations. The same motion
event cannot verify multiple edges. The current Habitat `ObservationPacket` has no such measurement,
so online Habitat runs cannot create verified traversal edges. Navigation task
outcome and physical passage counters are independent. The Habitat runner never
feeds evaluator success, oracle distance or goal position to memory; real
Habitat Episode outcomes therefore remain `unknown` until an agent observable
outcome source exists. Synthetic tests can supply a trusted outcome directly.

## Diagnostics and acceptance

Each V3 step has `v3_diagnostic` with stage counts for observed/persisted events,
three memory layers, four retrieval channels, selected/serialized prompt records,
reliability states, prompt hashes and token checks. `prompt_token_delta` is the
difference from applying the checkpoint tokenizer to the two human message
strings; `prompt_token_delta_basis` records this scope. The Engine separately
records final `input_ids` count/hash, the original instruction and memory block
presence, and the actual vLLM `prompt_token_ids` hash/presence. The evidence
IDs in `injected_evidence_ids` are reported only when their literal composite
keys occur in a memory record verified in the final token sequence (and in
`prompt_token_ids` for vLLM). `model_visible_evidence_ids` separately tracks
record-to-evidence associations even when an ID is not printed.
`historical_same_episode_prior_step` and `historical_cross_episode` use the
composite evidence identity; a bare `source_step < current_step` misses history
after step numbers reset. Per-channel historical candidate counts and the
separate `retrieval_nonempty`, `fusion_nonempty`, `formatter_nonempty`,
`actual_injection`, and `actual_historical_injection` fields expose each stage.
`prompt_historical_records_serialized` counts records whose printed Evidence ID
is historical; `prompt_historical_supported_records` also counts records whose
unprinted supporting evidence includes history. Only printed IDs can count as
literally visible in final tokens.
Candidate and serialized line lengths show what fits the 400-character budget.
If the Processor's final V3 prompt exceeds `model_max_length`, it raises before
truncation; baseline and legacy inputs keep their prior truncation behavior and
emit a length warning.

Run V3 only with `--rgm_v3_enabled` and an explicit new `--output_dir`. An
existing V3 directory can be reused only with `--resume` and matching protocol
marker. Do not place V3 results under `output/rgm_pilot_88_v1` or
`output/rgm_pilot_88_v2`. Do not compare V3 output with the old A/C/D reader;
that reader validates the immutable legacy Pilot protocol.

The Mem4Nav manuscript [arXiv:2506.19433](https://arxiv.org/abs/2506.19433)
is withdrawn. Its memory organization is an engineering reference only; this
project does not use its reported performance as evidence for V3.
