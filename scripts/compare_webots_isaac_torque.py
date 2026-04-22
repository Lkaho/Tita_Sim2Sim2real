#!/usr/bin/env python3

"""Compare the deployed Webots torque model with the Isaac Lab TITA task.

This script encodes the currently active formulas for the flat TITA policy:

- Webots / ros2_control deployment:
  - legs:   tau = kp * (q_des - q) + kd * (0 - dq)
  - wheels: tau = wheel_kp * (wheel_scale * action + wheel_offset) - wheel_kd * dq
            then apply the Webots overspeed brake: if |dq| >= 20 and tau * dq > 0 => tau = 0

- Isaac Lab training task DDT-Velocity-Flat-Tita-NoBaseVel-NoEstimator-v0:
  - legs:   tau = stiffness * (q_des - q) + damping * (0 - dq), then clip to +/- effort_limit
  - wheels: tau = wheel_effort_gain * (wheel_scale * action + wheel_offset) - wheel_damping * dq,
            then clip to +/- effort_limit

The action order on both sides is:
  [L1, L2, L3, L4, R1, R2, R3, R4]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


JOINT_NAMES = (
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
)

WHEEL_INDICES = {3, 7}


@dataclass(frozen=True)
class WebotsParams:
    default_joint_angles: tuple[float, ...] = (0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0)
    action_scales: tuple[float, ...] = (0.25, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5)
    joint_kp: tuple[float, ...] = (40.0, 40.0, 40.0, 11.5, 40.0, 40.0, 40.0, 11.5)
    joint_kd: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5)
    wheel_velocity_limit: float = 20.0


@dataclass(frozen=True)
class IsaacParams:
    default_joint_angles: tuple[float, ...] = (0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0)
    leg_scale: tuple[float, ...] = (0.25, 0.5, 0.5, 0.25, 0.5, 0.5)
    wheel_scale: float = 0.5
    wheel_effort_gain: float = 12.0
    leg_stiffness: float = 40.0
    leg_damping: float = 1.0
    wheel_damping: float = 0.8
    leg_effort_limit: float = 60.0
    wheel_effort_limit: float = 20.0
    wheel_velocity_limit: float = 20.0


def _parse_csv_floats(text: str, expected_len: int, name: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return values


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def compute_webots(actions: list[float], q: list[float], dq: list[float], params: WebotsParams) -> list[float]:
    tau = []
    for i in range(len(JOINT_NAMES)):
        scaled = actions[i] * params.action_scales[i]
        if i in WHEEL_INDICES:
            torque = params.joint_kp[i] * (scaled + params.default_joint_angles[i]) - params.joint_kd[i] * dq[i]
            if abs(dq[i]) >= params.wheel_velocity_limit and torque * dq[i] > 0.0:
                torque = 0.0
        else:
            q_des = scaled + params.default_joint_angles[i]
            torque = params.joint_kp[i] * (q_des - q[i]) + params.joint_kd[i] * (0.0 - dq[i])
        tau.append(torque)
    return tau


def compute_isaac(actions: list[float], q: list[float], dq: list[float], params: IsaacParams) -> list[float]:
    tau = [0.0] * len(JOINT_NAMES)

    leg_action_indices = (0, 1, 2, 4, 5, 6)
    for leg_slot, joint_index in enumerate(leg_action_indices):
        q_des = params.default_joint_angles[joint_index] + params.leg_scale[leg_slot] * actions[joint_index]
        torque = params.leg_stiffness * (q_des - q[joint_index]) + params.leg_damping * (0.0 - dq[joint_index])
        tau[joint_index] = _clip(torque, params.leg_effort_limit)

    for wheel_index in (3, 7):
        tau_ff = params.wheel_effort_gain * (params.wheel_scale * actions[wheel_index])
        torque = tau_ff + params.wheel_damping * (0.0 - dq[wheel_index])
        tau[wheel_index] = _clip(torque, params.wheel_effort_limit)

    return tau


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actions",
        default="0,0,0,1,0,0,0,1",
        help="8 comma-separated actions in [L1,L2,L3,L4,R1,R2,R3,R4] order.",
    )
    parser.add_argument(
        "--q",
        default="0,0.8,-1.5,0,0,0.8,-1.5,0",
        help="8 comma-separated joint positions.",
    )
    parser.add_argument(
        "--dq",
        default="0,0,0,5,0,0,0,5",
        help="8 comma-separated joint velocities.",
    )
    args = parser.parse_args()

    actions = _parse_csv_floats(args.actions, 8, "actions")
    q = _parse_csv_floats(args.q, 8, "q")
    dq = _parse_csv_floats(args.dq, 8, "dq")

    webots = compute_webots(actions, q, dq, WebotsParams())
    isaac = compute_isaac(actions, q, dq, IsaacParams())

    print("Joint                      webots_tau    isaac_tau     diff")
    print("---------------------------------------------------------------")
    for name, tau_webots, tau_isaac in zip(JOINT_NAMES, webots, isaac):
        diff = tau_webots - tau_isaac
        print(f"{name:24s} {tau_webots:11.6f} {tau_isaac:11.6f} {diff:11.6f}")


if __name__ == "__main__":
    main()
