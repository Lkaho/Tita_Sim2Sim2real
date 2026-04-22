// Copyright (c) 2026 Direct Drive Technology Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
constexpr double kWheelVelocityLimit = 20.0;
constexpr std::array<double, 8> kDefaultJointAngles = {0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0};
constexpr std::array<double, 8> kJointKp = {40.0, 40.0, 40.0, 11.5, 40.0, 40.0, 40.0, 11.5};
constexpr std::array<double, 8> kJointKd = {1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5};
constexpr std::array<double, 8> kActionScales = {0.25, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5};
constexpr std::array<double, 8> kTorqueLimit = {60.0, 60.0, 60.0, 20.0, 60.0, 60.0, 60.0, 20.0};

struct JointCommand
{
  double position = 0.0;
  double velocity = 0.0;
  double effort = 0.0;
  double kp = 0.0;
  double kd = 0.0;
};

bool is_wheel(size_t index)
{
  return index == 3 || index == 7;
}

double clamp_torque_command(double effort, size_t index)
{
  const double limit = std::abs(kTorqueLimit[index]);
  return std::clamp(effort, -limit, limit);
}

JointCommand command_from_action(double action, size_t index, double measured_velocity)
{
  const double action_scaled = action * kActionScales[index];
  const double command = action_scaled + kDefaultJointAngles[index];

  JointCommand joint_command;
  if (is_wheel(index)) {
    // Mirrors FSMState_RL::run() for wheel joints in control_type "P".
    joint_command.kp = 0.0;
    joint_command.kd = 0.0;
    joint_command.position = 0.0;
    joint_command.velocity = 0.0;
    joint_command.effort =
      clamp_torque_command(kJointKp[index] * command - kJointKd[index] * measured_velocity, index);
    return joint_command;
  }

  joint_command.kp = kJointKp[index];
  joint_command.kd = kJointKd[index];
  joint_command.position = command;
  joint_command.velocity = 0.0;
  joint_command.effort = 0.0;
  return joint_command;
}

double hardware_bridge_torque(
  const JointCommand & command, double measured_position, double measured_velocity)
{
  return command.effort + command.kp * (command.position - measured_position) +
         command.kd * (command.velocity - measured_velocity);
}

double apply_webots_velocity_limit(double effort, double measured_velocity, double velocity_limit)
{
  if (!std::isfinite(velocity_limit) || !std::isfinite(measured_velocity)) {
    return effort;
  }

  if (std::abs(measured_velocity) < velocity_limit) {
    return effort;
  }

  return effort * measured_velocity > 0.0 ? 0.0 : effort;
}

double webots_bridge_torque(
  const JointCommand & command, double measured_position, double measured_velocity,
  double velocity_limit = std::numeric_limits<double>::infinity())
{
  const double effort = command.kp * (command.position - measured_position) +
                        command.kd * (command.velocity - measured_velocity) + command.effort;
  return apply_webots_velocity_limit(effort, measured_velocity, velocity_limit);
}
}  // namespace

TEST(BridgeTorqueEquivalence, SameActionMatchesWhenWebotsVelocityLimitIsInactive)
{
  const std::array<double, 8> actions = {0.12, -0.35, 0.48, 0.586595, -0.2, 0.31, -0.17, 0.528778};
  const std::array<double, 8> positions = {0.01, 0.79, -1.48, -1.68, -0.02, 0.82, -1.55, -0.84};
  const std::array<double, 8> velocities = {0.1, -0.2, 0.05, 6.43464, -0.08, 0.12, -0.03, 6.39049};

  for (size_t i = 0; i < actions.size(); ++i) {
    const JointCommand command = command_from_action(actions[i], i, velocities[i]);

    const double hardware_torque = hardware_bridge_torque(command, positions[i], velocities[i]);
    const double webots_torque = webots_bridge_torque(
      command, positions[i], velocities[i],
      is_wheel(i) ? kWheelVelocityLimit : std::numeric_limits<double>::infinity());

    EXPECT_NEAR(hardware_torque, webots_torque, 1e-12) << "joint index " << i;
  }
}

TEST(BridgeTorqueEquivalence, WebotsWheelVelocityLimitCanMakeTorqueDifferent)
{
  const size_t wheel_index = 3;
  const double action = 5.0;
  const double position = 0.0;
  const double velocity_over_limit = 25.0;
  const JointCommand command = command_from_action(action, wheel_index, velocity_over_limit);

  const double hardware_torque =
    hardware_bridge_torque(command, position, velocity_over_limit);
  const double webots_torque =
    webots_bridge_torque(command, position, velocity_over_limit, kWheelVelocityLimit);

  ASSERT_GT(hardware_torque * velocity_over_limit, 0.0);
  EXPECT_DOUBLE_EQ(webots_torque, 0.0);
  EXPECT_NE(hardware_torque, webots_torque);
}
