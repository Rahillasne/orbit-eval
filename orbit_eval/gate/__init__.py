# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright 2026 ORBIT Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""orbit_eval.gate — the Ship-Gate subpackage (v0, FIXED-SAMPLE).

A statistical promotion gate for robot policies: CRN pre-checks, the paired
verdict engine (exact McNemar + episode bootstrap), winner's-curse handling,
and a signed, auditable decision record appended to a JSONL ledger.

v0 is fixed-sample: one look at a pre-sized eval. The anytime-valid
SEQUENTIAL layer (peek-as-you-go, Ville-bounded) is v1: gate.sequential.

Pure re-exports, no logic: records (schema / config / hashing / ledger),
engine (the verdict engine), replay (historical corpus replay),
sequential (the v1 e-process streams), shadow (the WS3 capture +
shadow-gating layer over a sha256-chained ledger).
"""

from .records import (          # noqa: F401
    SCHEMA_VERSION,
    canonical_json,
    sha256_hex,
    GateConfig,
    build_record,
    record_hash,
    append_record,
    iter_ledger,
    verify_ledger,
)
from .engine import (           # noqa: F401
    GateDecision,
    GateInputError,
    gate,
    resolve_sigma_run,
    CAVEAT_FLOOR_2PP,
    curse_from_prior,
    curse_from_history,
    episodes_for_min_effect,
    power_at_min_effect_paired,
    episodes_for_min_effect_unpaired,
    power_at_min_effect_unpaired,
)
from .sequential import (       # noqa: F401
    DECISION_TO_VERDICT,
    EpisodeSequential,
    RetrainSequential,
)
from .replay import (           # noqa: F401
    CorpusRun,
    parse_corpus_line,
    load_corpus,
    stack_epoch,
    find_null_pairs,
    retrain_prior_regime,
    replay_null_pairs,
    replay_curse,
    replay_effects,
    render_report,
    run_replay,
)
from .shadow import (           # noqa: F401
    GENESIS_SHA256,
    SHADOW_SCHEMA_VERSION,
    content_sha256,
    read_chain,
    append_chained,
    verify_chain,
    capture_tree,
    latest_capture,
    eval_run_from_capture,
    matched_config_mismatch,
    shadow_pair,
    shadow_report,
)

__all__ = [
    # records
    "SCHEMA_VERSION",
    "canonical_json",
    "sha256_hex",
    "GateConfig",
    "build_record",
    "record_hash",
    "append_record",
    "iter_ledger",
    "verify_ledger",
    # engine
    "GateDecision",
    "GateInputError",
    "gate",
    "resolve_sigma_run",
    "CAVEAT_FLOOR_2PP",
    "curse_from_prior",
    "curse_from_history",
    "episodes_for_min_effect",
    "power_at_min_effect_paired",
    "episodes_for_min_effect_unpaired",
    "power_at_min_effect_unpaired",
    # sequential (v1)
    "DECISION_TO_VERDICT",
    "EpisodeSequential",
    "RetrainSequential",
    # replay
    "CorpusRun",
    "parse_corpus_line",
    "load_corpus",
    "stack_epoch",
    "find_null_pairs",
    "retrain_prior_regime",
    "replay_null_pairs",
    "replay_curse",
    "replay_effects",
    "render_report",
    "run_replay",
    # shadow (WS3)
    "GENESIS_SHA256",
    "SHADOW_SCHEMA_VERSION",
    "content_sha256",
    "read_chain",
    "append_chained",
    "verify_chain",
    "capture_tree",
    "latest_capture",
    "eval_run_from_capture",
    "matched_config_mismatch",
    "shadow_pair",
    "shadow_report",
]
