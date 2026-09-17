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

"""orbit-eval log — record trials at the robot, one keypress each.

Everything else in this package reads evaluations that already exist. On a real
robot they do not exist: somebody stands at the cell, runs the policy, watches,
and writes a tick on paper. That tally is the most expensive data in robotics and
it is routinely lost, mistyped, or kept in a spreadsheet nobody can join to a
checkpoint.

So this is the wedge, and it is deliberately the least clever file here:

  y  the trial succeeded          u  undo the last one
  n  it failed, then one key for why    s  skip (reset failed, not the policy's fault)
  q  done

Every keypress is appended to the CSV before the next one is read, so killing the
terminal, losing the ssh connection or running out of battery costs you at most
the trial you were watching. Re-running the same command resumes where the file
left off rather than starting a second tally.

Two fields exist because of what they let the rest of the tool do later, not
because they are nice to have:

  block   which robot, cell, day or operator this trial ran on. `route` already
          stratifies by block, so with it you can ask whether a regression showed
          up on every robot or only on cell 3. Without it that question is
          unanswerable after the fact.
  reason  one keypress of failure taxonomy. A success rate says a job is at 45%;
          a reason column says 30 of the 55 failures were the grasp. That is the
          difference between "collect more demonstrations" and "collect more
          demonstrations OF THE GRASP", and it is the gap `next` currently admits
          it cannot fill.

The columns are the ones `orbit check` already reads, so the file this writes is
an input to the rest of the tool with no conversion step.
"""

import csv
import os
import sys
import time

from . import nextstep
from . import route

HEADER = ["version", "skill", "episode", "success", "reason", "block", "started_utc"]

REASONS = [
    ("g", "grasp", "picked it up wrong, or not at all"),
    ("p", "place", "got it, put it in the wrong place"),
    ("a", "approach", "never reached the object"),
    ("m", "moved", "object or scene moved, setup problem"),
    ("t", "timeout", "ran out of time without finishing"),
    ("c", "collision", "hit something"),
    ("x", "other", "none of the above"),
]
REASON_KEYS = {k: name for k, name, _ in REASONS}


class Session(object):
    """The tally for one (policy, job), backed by a CSV that is always current.

    Pure enough to test without a terminal: `record`, `undo` and `rows` know
    nothing about keypresses.
    """

    def __init__(self, path, policy, job, block=""):
        self.path = path
        self.policy = policy
        self.job = job
        self.block = block
        self.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.rows = []
        self.other_rows = []
        self._load()

    # ---------------------------------------------------------------- storage

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("version") == self.policy and r.get("skill") == self.job:
                    self.rows.append(r)
                else:
                    self.other_rows.append(r)
        self.rows.sort(key=lambda r: int(r.get("episode") or 0))

    def _flush(self):
        """Rewrite the whole file. Tallies are small and correctness beats cleverness."""
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HEADER, extrasaction="ignore")
            w.writeheader()
            for r in self.other_rows + self.rows:
                w.writerow({k: r.get(k, "") for k in HEADER})
        os.replace(tmp, self.path)

    def _append(self, row):
        """Append one row without rewriting, so a crash costs one trial at most."""
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HEADER, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in HEADER})
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------- tally

    def record(self, success, reason=""):
        row = {"version": self.policy, "skill": self.job, "episode": len(self.rows),
               "success": 1 if success else 0, "reason": "" if success else reason,
               "block": self.block, "started_utc": self.started}
        self.rows.append(row)
        if self.other_rows or len(self.rows) == 1:
            # a shared file, or the very first row: rewrite so ordering stays sane
            self._flush()
        else:
            self._append(row)
        return row

    def undo(self):
        if not self.rows:
            return None
        gone = self.rows.pop()
        self._flush()
        return gone

    # ------------------------------------------------------------------ stats

    @property
    def n(self):
        return len(self.rows)

    @property
    def k(self):
        return sum(1 for r in self.rows if str(r.get("success")) in ("1", "True", "true"))

    @property
    def rate(self):
        return 100.0 * self.k / self.n if self.n else float("nan")

    def ci(self):
        # points, matching every other rate in this tool
        return route.wilson(self.k, self.n) if self.n else (float("nan"), float("nan"))

    def reasons(self):
        out = {}
        for r in self.rows:
            if str(r.get("success")) in ("1", "True", "true"):
                continue
            key = r.get("reason") or "unrecorded"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def target_n(self, target_pp=nextstep.TARGET_PP):
        return nextstep.episodes_for(self.rate if self.n else 50.0, target_pp)


# ------------------------------------------------------------------ terminal

def _getch():
    """One keypress, or a whole line when stdin is not a terminal (tests, pipes)."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "q"
        return (line.strip() or "\n")[:1].lower()
    try:
        import termios
        import tty
    except ImportError:                                    # Windows
        try:
            import msvcrt
            return msvcrt.getch().decode("utf-8", "replace").lower()
        except Exception:
            return (sys.stdin.readline().strip() or "\n")[:1].lower()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\x03", "\x04"):                             # ctrl-c, ctrl-d
        return "q"
    return ch.lower()


def prompt(s, target_pp=nextstep.TARGET_PP):
    """The two lines the operator reads between trials."""
    if s.n:
        lo, hi = s.ci()
        so_far = "%d/%d = %.0f%%   95%% %.0f-%.0f%%" % (s.k, s.n, s.rate, lo, hi)
    else:
        so_far = "nothing recorded yet"
    need = s.target_n(target_pp)
    left = max(0, (need or 0) - s.n)
    tail = ("  %d more to call a %.0f-point drop" % (left, target_pp)) if left else \
           ("  enough trials to call a %.0f-point drop" % target_pp)
    head = "%s / %s%s   trial %d" % (s.job, s.policy,
                                     (" / " + s.block) if s.block else "", s.n + 1)
    return ("\n  %s\n  %s\n  %s%s\n\n  [y] success   [n] failure   [u] undo   "
            "[s] skip   [q] done\n" % (head, "-" * max(len(head), 34), so_far, tail))


def reason_prompt():
    keys = "   ".join("[%s] %s" % (k, name) for k, name, _ in REASONS)
    return "  why did it fail?  %s   [enter] skip\n" % keys


def summary(s, target_pp=nextstep.TARGET_PP):
    L = ["", "  %s / %s" % (s.job, s.policy), "  " + "-" * 40]
    if not s.n:
        L.append("  nothing recorded.")
        return "\n".join(L)
    lo, hi = s.ci()
    L.append("  %d trials, %d succeeded = %.0f%%   95%% interval %.0f-%.0f%%"
             % (s.n, s.k, s.rate, lo, hi))
    rs = s.reasons()
    if rs:
        L.append("  failures: " + ", ".join("%s %d" % (k, v) for k, v in rs.items()))
    need = s.target_n(target_pp)
    if need and s.n < need:
        L.append("  At %d trials the smallest drop you could call is wider than %.0f points."
                 % (s.n, target_pp))
        L.append("  %d more would get you there." % (need - s.n))
    else:
        L.append("  Enough trials here to call a %.0f-point drop." % target_pp)
    L.append("")
    L.append("  saved to %s" % s.path)
    L.append("  run `orbit check` when every policy has been through every job.")
    return "\n".join(L)


def loop(s, out=None, target_pp=nextstep.TARGET_PP, getch=_getch):
    """Read keys until quit. Returns the session."""
    out = out or sys.stdout
    while True:
        out.write(prompt(s, target_pp))
        out.flush()
        ch = getch()
        if ch in ("q", "\x1b"):
            break
        if ch == "y":
            s.record(True)
            out.write("  recorded: success\n")
        elif ch == "n":
            out.write(reason_prompt())
            out.flush()
            rk = getch()
            reason = REASON_KEYS.get(rk, "")
            s.record(False, reason)
            out.write("  recorded: failure%s\n" % (" (%s)" % reason if reason else ""))
        elif ch == "u":
            gone = s.undo()
            out.write("  undone: %s\n" % ("nothing to undo" if gone is None else
                                          ("success" if str(gone.get("success")) == "1"
                                           else "failure")))
        elif ch == "s":
            out.write("  skipped, nothing recorded\n")
        else:
            out.write("  press y, n, u, s or q\n")
        out.flush()
    out.write(summary(s, target_pp) + "\n")
    out.flush()
    return s
