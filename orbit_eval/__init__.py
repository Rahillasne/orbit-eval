# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright (C) 2026 ORBIT Research
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the license along with this program. If not,
# see <https://www.gnu.org/licenses/>.
#
# A commercial license, exempting you from the AGPL's source-disclosure terms, is
# available from ORBIT Research.

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

__version__ = "0.8.1"

from . import atlas, power, stats          # noqa: F401
from .io import EvalRun, parse_file        # noqa: F401
