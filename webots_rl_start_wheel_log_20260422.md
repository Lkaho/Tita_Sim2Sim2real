# Webots RL start wheel log - 2026-04-22

Command:

```bash
ros2 launch rl_controller sim_webots.launch.py robot:=tita terrain:=tita
ros2 run keyboard_controller keyboard_controller_node
```

Key sequence:

1. Pressed `7` to enter `transform_up`.
2. Waited a few seconds.
3. Pressed `0` to enter `rl_flat`.

Policy used by this run:

```text
config/tita/policy_flat.onnx
```

Note:

- The values below are from `[FSMState_RL][wheel_debug]`.
- Webots wheel velocity limit is `20 rad/s`.
- During these first RL samples, `abs(dq)` is below `20 rad/s`, so the Webots velocity limiter should not zero the torque.
- For wheel joints, `low_kp=0` and `low_kd=0`, so the Webots applied torque should be the same as `final_tau` here.

## Last sample before entering rl_flat

| sample | idx | raw_action | scaled | dq rad/s | final_tau Nm |
|---:|---:|---:|---:|---:|---:|
| pre-RL | 3 | 0.000000 | 0.000000 | 0.000407 | -0.000326 |
| pre-RL | 7 | 0.000000 | 0.000000 | 0.000397 | -0.000317 |

## First samples after entering rl_flat

| sample | idx | raw_action | scaled | dq rad/s | final_tau Nm |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | -0.490506 | -0.245253 | -5.448580 | 1.538460 |
| 1 | 7 | -0.460502 | -0.230251 | -5.245500 | 1.548510 |
| 2 | 3 | -0.796191 | -0.398096 | -3.822240 | -1.520310 |
| 2 | 7 | -0.697364 | -0.348682 | -2.975670 | -1.629300 |
| 3 | 3 | -0.442533 | -0.221267 | -0.165260 | -2.412360 |
| 3 | 7 | -0.388043 | -0.194022 | 0.008453 | -2.238010 |
| 4 | 3 | 0.469496 | 0.234748 | 2.983110 | 0.313116 |
| 4 | 7 | 0.399553 | 0.199777 | 2.588950 | 0.226269 |
| 5 | 3 | 0.350389 | 0.175195 | 1.576480 | 0.753550 |
| 5 | 7 | 0.295082 | 0.147541 | 1.326780 | 0.635295 |
| 6 | 3 | 0.063981 | 0.031990 | -0.123194 | 0.466446 |
| 6 | 7 | 0.029017 | 0.014508 | -0.170656 | 0.303371 |
| 7 | 3 | 0.034918 | 0.017459 | 0.113878 | 0.109674 |
| 7 | 7 | 0.003058 | 0.001529 | 0.074503 | -0.042018 |
| 8 | 3 | 0.027656 | 0.013828 | 0.108733 | 0.072036 |
| 8 | 7 | -0.003697 | -0.001849 | 0.074152 | -0.080581 |

## Quick reading

- The first RL sample is not huge in Webots: wheel torque is about `+1.54 Nm` on both wheels.
- The largest wheel speed in these first samples is about `5.45 rad/s`, far below the `20 rad/s` limiter.
- The largest wheel torque in these first samples is about `2.41 Nm`, also far below the `20 Nm` clamp.
- This Webots run does not reproduce the real-machine issue where the wheel torque kicks hard at RL entry.

