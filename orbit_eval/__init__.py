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

"""orbit-eval — honest statistics for robot-policy evaluation.

A thin, pip-installable port of the validated statistics from the ORBIT
research program, shipping the measured noise atlas as data. It always
distinguishes:

  (a) fixed-checkpoint / fixed-dataset comparisons under common random
      numbers — cheap, powered at ~2 seeds against the sigma_0 harness
      floor; from
  (b) selection-method comparisons across independent retrains — dominated
      by sigma_run and sigma_set, needing ~59 set draws per arm for a
      10 pp effect in the LIBERO product regime.

CRN pairing cannot rescue (b): pairing cancels eval-draw noise only, while
sigma_run and sigma_set live in training.
"""

__version__ = "0.8.3"

from . import atlas, power, stats          # noqa: F401
from .io import EvalRun, parse_file        # noqa: F401
