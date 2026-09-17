# Bundled demo data

Two public LeRobot datasets, metadata only. No frames, no video, no parquet.
Both are Apache-2.0 on the Hugging Face Hub and are redistributed unchanged
except where noted. `orbit status --demo` and `orbit cover --demo` read these so
the tool can be tried on a machine with no data on it.

## `so101_pickplace`

`lerobot/svla_so101_pickplace`, the official SO-101 example dataset.
https://huggingface.co/datasets/lerobot/svla_so101_pickplace

- `meta/info.json` and `meta/stats.json`, fetched 2026-09-17, byte-identical to
  the Hub copies.
- v3.0 layout, 50 episodes, 11,939 frames at 30 fps, one task.
- Two action bounds sit exactly on the normalised limit (`shoulder_lift.pos` at
  -100, `elbow_flex.pos` at +100). See
  `validation/NORMALISATION_BOUND_2026-09-17.md` for why that is a clipped
  command and not a rescaling artefact.

## `so101_table_cleanup`

`youliangtan/so101-table-cleanup`, a community SO-101 recording.
https://huggingface.co/datasets/youliangtan/so101-table-cleanup

- `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, fetched
  2026-09-17, byte-identical to the Hub copies.
- `meta/episodes_stats.jsonl`: the per-image statistics
  (`observation.images.*`) were removed to keep the package small. Every other
  key is unchanged. The Hub file is 176 KB; this one is 116 KB.
- v2.1 layout, 80 episodes, four instructions of the shape
  "Grab {X} and place into pen holder".
