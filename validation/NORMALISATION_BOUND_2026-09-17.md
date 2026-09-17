# Is -100..100 the calibration bound, or a per-dataset rescaling?

**Answer: the calibration bound. The clipping finding stands.**

Checked 2026-09-17 against two copies of LeRobot: the installed `lerobot` 0.5.0
from PyPI and a clone of `huggingface/lerobot` `main` at `30074f7` (2026-09-16).
The lines below are identical in both.

## What the code does

`lerobot/motors/motors_bus.py`, `MotorsBus._normalize` (main: lines 854 to 880):

```python
min_ = self.calibration[motor].range_min
max_ = self.calibration[motor].range_max
...
bounded_val = min(max_, max(min_, val))
if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
    norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
    norm = ((bounded_val - min_) / (max_ - min_)) * 100
```

`range_min` and `range_max` are the per-motor encoder positions recorded when the
user ran `lerobot-calibrate` and swept each joint to its limits. They are stored
in the calibration file, not derived from any dataset. The raw reading is clamped
to that range **before** it is mapped, so a normalised value of exactly -100.0 or
+100.0 means the encoder was at or beyond the calibrated limit and the value was
clipped. Nothing in the dataset writer rescales these numbers afterwards.

`_unnormalize` (lines 896 to 903) clamps the other way: a command outside
[-100, 100] is clipped to the bound before it reaches the motor. A policy trained
on clipped frames therefore learns commands the arm cannot execute, which is what
`orbit status` says.

## Which joints use which scale

`lerobot/robots/so_follower/so_follower.py` lines 50 and 59, and the matching
leader at `teleoperators/so_leader/so_leader.py` lines 42 and 51:

- the five arm joints: `RANGE_M100_100` (or `DEGREES` when `use_degrees=True`)
- the gripper: `RANGE_0_100`

The recorded `action` is the leader arm's `get_action()` (so_leader.py line 146),
which returns these normalised values directly. So a dataset action at exactly
-100 or +100 on an arm joint is a clipped leader reading.

## What this means for the check in `orbit_eval/dataset.py`

- **Arm joints at -100 or +100**: clipped at the calibration bound. Report it.
- **Gripper at 0 or 100**: the calibrated closed and open positions. A gripper
  sits at its bound whenever it closes on nothing or opens fully, so a bound hit
  is ordinary behaviour, not a defect. The gripper is now exempt at both ends.
- **`use_degrees=True` datasets**: `DEGREES` mode has no clamp and its values are
  in degrees. `_looks_normalised` already leaves any dataset whose values fall
  outside the -100..100 scale alone, so these are not checked.

## The finding on the official example

`lerobot/svla_so101_pickplace` (Apache-2.0, v3.0, 50 episodes): `action.min[1]`
(`shoulder_lift.pos`) is exactly -100.0 and `action.max[2]` (`elbow_flex.pos`) is
exactly 100.0. Two of ten arm-joint bounds sit on the limit; the other eight sit
between -93.5 and 99.5, so this is not min-max normalisation making the check
vacuous. Both values are in `meta/stats.json` on the Hub and are bundled with
this package for `orbit status --demo`.
