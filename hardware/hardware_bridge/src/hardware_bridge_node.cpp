// Copyright (c) 2023 Direct Drive Technology Co., Ltd. All rights reserved.
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

#include "hardware_bridge/hardware_bridge_node.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <pthread.h>
#include <sched.h>

#include "pluginlib/class_list_macros.hpp"

namespace tita_locomotion
{
namespace
{
constexpr double kWheelVelocityLimit = 20.0;
constexpr double kWheelTorqueLimit = 20.0;
constexpr double kWheelTorqueSlewRate = 120.0;  // Nm/s
constexpr double kFallbackControlDt = 0.002;    // 500 Hz ros2_control loop

bool is_wheel_joint(size_t id, size_t leg_dof)
{
  return leg_dof > 0 && id % leg_dof == leg_dof - 1;
}

double apply_wheel_velocity_limit(double torque, double velocity)
{
  if (!std::isfinite(torque) || !std::isfinite(velocity)) {
    return torque;
  }

  if (std::abs(velocity) < kWheelVelocityLimit) {
    return torque;
  }

  return torque * velocity > 0.0 ? 0.0 : torque;
}

double apply_torque_slew_limit(double target, double previous, double max_delta)
{
  if (!std::isfinite(target) || !std::isfinite(previous) || max_delta <= 0.0) {
    return target;
  }

  return previous + std::clamp(target - previous, -max_delta, max_delta);
}
}  // namespace

HardwareBridge::HardwareBridge() {}
HardwareBridge::~HardwareBridge()
{
  // if (direct_mode_) {
  //   std::vector<double> cmd_torque;
  //   cmd_torque.resize(mJoints.size(), 0);
  //   if (!robot_->set_target_joint_t(cmd_torque)) {
  //     RCLCPP_ERROR(
  //       rclcpp::get_logger("hardware_bridge"), "Failed to set target joint torque on exit");
  //   }
  // }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn HardwareBridge::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  // node_ = rclcpp::Node::make_shared("hardware_bridge_node");
  // executor_.add_node(node_);
  // std::thread([this]() { executor_.spin(); }).detach();
  auto ctrl_mode = info_.hardware_parameters["pvt_ctrl"];
  pvt_ctrl_ = ctrl_mode.compare("false") == 0 ? false : true;
  RCLCPP_INFO(rclcpp::get_logger("hardware_bridge"), "Using PVT control: %d", pvt_ctrl_);
  for (hardware_interface::ComponentInfo component : info.joints) {
    Joint joint;
    joint.name = component.name;
    mJoints.push_back(joint);
  }
  for (hardware_interface::ComponentInfo component : info.sensors) {
    if (component.name.find("_imu") != std::string::npos) {
      mImu.name = component.name;
    }
  }

  robot_ = std::make_unique<tita_robot>(mJoints.size());  // depends on urdf
  if (mJoints.size() % 8 == 0)
    leg_dof_ = 4;
  else if (mJoints.size() % 6 == 0)
    leg_dof_ = 3;
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS) {
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::ERROR;
  }
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> HardwareBridge::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (Joint & joint : mJoints) {
    interfaces.emplace_back(
      hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_POSITION, &(joint.position)));
    interfaces.emplace_back(
      hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_VELOCITY, &(joint.velocity)));
    interfaces.emplace_back(
      hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_EFFORT, &(joint.effort)));
  }

  for (hardware_interface::ComponentInfo component : info_.sensors) {
    if (component.name == mImu.name) {
      for (uint i = 0; i < 4; i++) {
        interfaces.emplace_back(
          hardware_interface::StateInterface(
            component.name, component.state_interfaces[i].name, &mImu.orientation[i]));
      }
      for (uint i = 0; i < 3; i++) {
        interfaces.emplace_back(
          hardware_interface::StateInterface(
            component.name, component.state_interfaces[i + 4].name, &mImu.angular_velocity[i]));
      }
      for (uint i = 0; i < 3; i++) {
        interfaces.emplace_back(
          hardware_interface::StateInterface(
            component.name, component.state_interfaces[i + 7].name, &mImu.linear_acceleration[i]));
      }
    }
  }

  return interfaces;
}

std::vector<hardware_interface::CommandInterface> HardwareBridge::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (Joint & joint : mJoints) {
    interfaces.emplace_back(
      hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_POSITION, &(joint.positionCommand)));
    interfaces.emplace_back(
      hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_VELOCITY, &(joint.velocityCommand)));
    interfaces.emplace_back(
      hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_EFFORT, &(joint.effortCommand)));
    interfaces.emplace_back(hardware_interface::CommandInterface(joint.name, "kp", &(joint.kp)));
    interfaces.emplace_back(hardware_interface::CommandInterface(joint.name, "kd", &(joint.kd)));
  }
  return interfaces;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HardwareBridge::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HardwareBridge::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type HardwareBridge::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (robot_->imu_data_timeout()) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("hardware_bridge"), clock_, 10000,
      "Imu data read timeout, waiting for connect");

  } else if (robot_->motors_data_timeout()) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("hardware_bridge"), clock_, 10000,
      "Motors data read timeout, waiting for connect");
  }

  // read from can bus
  auto q = robot_->get_joint_q();
  auto v = robot_->get_joint_v();
  auto t = robot_->get_joint_t();
  auto status = robot_->get_joint_status();
  for (size_t id = 0; id < mJoints.size(); id++) {
    mJoints[id].position = q[id];
    mJoints[id].velocity = v[id];
    mJoints[id].effort = t[id];
    if (id % leg_dof_ == 1) {
      if (mJoints[id].position < -2.5) {
        mJoints[id].position += 2 * M_PI;
      }
    }
    mJoints[id].errorId = status[id];
  }
  auto quat = robot_->get_imu_quaternion();
  auto accl = robot_->get_imu_acceleration();
  auto gyro = robot_->get_imu_angular_velocity();

  for (size_t id = 0; id < 3; id++) {
    mImu.linear_acceleration[id] = accl[id];
    mImu.angular_velocity[id] = gyro[id];
    mImu.orientation[id] = quat[id];
  }

  mImu.orientation[3] = quat[3];
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type HardwareBridge::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  std::vector<double> kp, kd, p, v, t, only_ts;
  if (last_only_ts_.size() != mJoints.size()) {
    last_only_ts_.assign(mJoints.size(), 0.0);
    has_last_only_ts_ = false;
  }

  const double period_seconds = period.seconds();
  const double dt = period_seconds > 0.0 && std::isfinite(period_seconds)
                      ? period_seconds
                      : kFallbackControlDt;
  const double max_wheel_torque_delta = kWheelTorqueSlewRate * dt;
  std::vector<std::string> wheel_debug_entries;

  for (size_t id = 0; id < mJoints.size(); id++) {
    kp.push_back(mJoints[id].kp);
    kd.push_back(mJoints[id].kd);
    if (id % leg_dof_ == 1 && mJoints[id].positionCommand > M_PI) {
      mJoints[id].positionCommand -= 2 * M_PI;
    }
    p.push_back(mJoints[id].positionCommand);
    v.push_back(mJoints[id].velocityCommand);
    t.push_back(mJoints[id].effortCommand);
    const auto raw_only_t =
      mJoints[id].effortCommand +
      mJoints[id].kp * (mJoints[id].positionCommand - mJoints[id].position) +
      mJoints[id].kd * (mJoints[id].velocityCommand - mJoints[id].velocity);

    auto only_t = raw_only_t;
    auto velocity_limited_t = raw_only_t;
    if (is_wheel_joint(id, leg_dof_)) {
      const auto torque_limited_t =
        std::clamp(raw_only_t, -kWheelTorqueLimit, kWheelTorqueLimit);
      velocity_limited_t = apply_wheel_velocity_limit(torque_limited_t, mJoints[id].velocity);
      only_t = has_last_only_ts_
                 ? apply_torque_slew_limit(
                     velocity_limited_t, last_only_ts_[id], max_wheel_torque_delta)
                 : velocity_limited_t;

      std::ostringstream entry;
      entry << " {idx=" << id << " q=" << mJoints[id].position
            << " dq=" << mJoints[id].velocity << " tau_ff=" << mJoints[id].effortCommand
            << " kp=" << mJoints[id].kp << " kd=" << mJoints[id].kd
            << " qd=" << mJoints[id].positionCommand
            << " qd_dot=" << mJoints[id].velocityCommand << " raw=" << raw_only_t
            << " torque_limited=" << torque_limited_t
            << " vel_limited=" << velocity_limited_t << " final=" << only_t << "}";
      wheel_debug_entries.push_back(entry.str());
    }

    last_only_ts_[id] = only_t;
    only_ts.push_back(only_t);
  }
  has_last_only_ts_ = true;

  if (!wheel_debug_entries.empty()) {
    std::ostringstream msg;
    msg << "[HardwareBridge][wheel_protect]";
    for (const auto & entry : wheel_debug_entries) {
      msg << entry;
    }
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("hardware_bridge"), clock_, 500, "%s", msg.str().c_str());
  }

  if (pvt_ctrl_) {
    if (!robot_->set_target_joint_mit(p, v, kp, kd, t)) {
      RCLCPP_ERROR(rclcpp::get_logger("hardware_bridge"), "Failed to set target PVT on write");
      // return hardware_interface::return_type::ERROR;
    }
  } else {
    if (!robot_->set_target_joint_t(only_ts)) {
      RCLCPP_ERROR(rclcpp::get_logger("hardware_bridge"), "Failed to set target T on write");
      // return hardware_interface::return_type::ERROR;
    }
  }

  return hardware_interface::return_type::OK;
}

}  // namespace tita_locomotion
PLUGINLIB_EXPORT_CLASS(tita_locomotion::HardwareBridge, hardware_interface::SystemInterface)
