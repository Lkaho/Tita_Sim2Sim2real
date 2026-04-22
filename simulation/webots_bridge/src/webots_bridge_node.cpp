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

#include "webots_bridge/webots_bridge_node.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "webots/device.h"
#include "webots/robot.h"
namespace tita_webots_ros2_control
{
namespace
{
constexpr double kWheelVelocityLimit = 20.0;

WbDeviceTag find_device_by_name(const std::string & name, WbNodeType expected_type)
{
  if (name.empty()) {
    return 0;
  }

  WbDeviceTag device = wb_robot_get_device(name.c_str());
  if (device != 0 && wb_device_get_node_type(device) == expected_type) {
    return device;
  }
  return 0;
}

WbDeviceTag find_first_device_by_type(WbNodeType expected_type)
{
  const int device_count = wb_robot_get_number_of_devices();
  for (int i = 0; i < device_count; ++i) {
    WbDeviceTag device = wb_robot_get_device_by_index(i);
    if (device != 0 && wb_device_get_node_type(device) == expected_type) {
      return device;
    }
  }
  return 0;
}

std::string device_name_or_empty(WbDeviceTag device)
{
  if (device == 0) {
    return "";
  }

  const char * name = wb_device_get_name(device);
  return name == nullptr ? "" : std::string(name);
}

bool is_wheel_joint_name(const std::string & joint_name)
{
  return joint_name.find("_leg_4") != std::string::npos;
}

double apply_velocity_limit(const Joint & joint, double effort)
{
  if (!std::isfinite(joint.velocityLimit) || !std::isfinite(joint.velocity)) {
    return effort;
  }

  if (std::abs(joint.velocity) < joint.velocityLimit) {
    return effort;
  }

  // Match the actuator semantics used in Isaac Lab: once the wheel is already over the
  // configured speed limit, only allow torques that brake it back toward the limit.
  return effort * joint.velocity > 0.0 ? 0.0 : effort;
}
}  // namespace

WebotsBridge::WebotsBridge() { mNode = NULL; }
void WebotsBridge::init(
  webots_ros2_driver::WebotsNode * node, const hardware_interface::HardwareInfo & info)
{
  mNode = node;
  for (hardware_interface::ComponentInfo component : info.joints) {
    Joint joint;
    joint.name = component.name;
    if (is_wheel_joint_name(joint.name)) {
      joint.velocityLimit = kWheelVelocityLimit;
    }

    WbDeviceTag device = wb_robot_get_device(joint.name.c_str());
    WbNodeType type = wb_device_get_node_type(device);
    joint.motor = (type == WB_NODE_LINEAR_MOTOR || type == WB_NODE_ROTATIONAL_MOTOR)
                    ? device
                    : wb_position_sensor_get_motor(device);
    device = (component.parameters.count("sensor") == 0)
               ? wb_robot_get_device(joint.name.c_str())
               : wb_robot_get_device(component.parameters.at("sensor").c_str());
    type = wb_device_get_node_type(device);
    joint.sensor =
      (type == WB_NODE_POSITION_SENSOR) ? device : wb_motor_get_position_sensor(device);

    if (joint.sensor) {
      wb_position_sensor_enable(joint.sensor, wb_robot_get_basic_time_step());
      wb_motor_enable_torque_feedback(joint.motor, wb_robot_get_basic_time_step());
    }
    if (!joint.sensor && !joint.motor) {
      throw std::runtime_error("Cannot find a Motor or PositionSensor with name " + joint.name);
    }
    // Check if state interfaces have initial positions
    for (hardware_interface::InterfaceInfo stateInterface : component.state_interfaces) {
      if (stateInterface.name == "position" && !stateInterface.initial_value.empty()) {
        joint.position = std::stod(stateInterface.initial_value);
        wb_motor_set_position(joint.motor, std::stod(stateInterface.initial_value));
      }
    }

    if (std::isfinite(joint.velocityLimit)) {
      RCLCPP_INFO(
        rclcpp::get_logger("webots_bridge"), "Applied velocity limit %.2f rad/s to joint '%s'",
        joint.velocityLimit, joint.name.c_str());
    }

    mJoints.push_back(joint);
  }
  wb_robot_step(wb_robot_get_basic_time_step());

  for (hardware_interface::ComponentInfo component : info.sensors) {
    std::string sensor_name = component.name;
    WbDeviceTag inertial_device =
      find_device_by_name(sensor_name + " inertial", WB_NODE_INERTIAL_UNIT);
    if (inertial_device == 0) {
      inertial_device = find_device_by_name(sensor_name, WB_NODE_INERTIAL_UNIT);
    }
    if (inertial_device == 0) {
      inertial_device = find_first_device_by_type(WB_NODE_INERTIAL_UNIT);
    }

    if (inertial_device != 0) {
      mImu.name = sensor_name;
      mImu.inertialUnit = inertial_device;
      mImu.gyro = find_device_by_name(sensor_name + " gyro", WB_NODE_GYRO);
      if (mImu.gyro == 0) {
        mImu.gyro = find_first_device_by_type(WB_NODE_GYRO);
      }
      mImu.accelerometer = find_device_by_name(
        sensor_name + " accelerometer", WB_NODE_ACCELEROMETER);
      if (mImu.accelerometer == 0) {
        mImu.accelerometer = find_first_device_by_type(WB_NODE_ACCELEROMETER);
      }

      wb_inertial_unit_enable(mImu.inertialUnit, wb_robot_get_basic_time_step());
      if (mImu.gyro) wb_gyro_enable(mImu.gyro, wb_robot_get_basic_time_step());
      if (mImu.accelerometer)
        wb_accelerometer_enable(mImu.accelerometer, wb_robot_get_basic_time_step());

      RCLCPP_INFO(
        rclcpp::get_logger("webots_bridge"),
        "Mapped ros2_control IMU '%s' to Webots devices: inertial='%s', gyro='%s', accelerometer='%s'",
        sensor_name.c_str(), device_name_or_empty(mImu.inertialUnit).c_str(),
        device_name_or_empty(mImu.gyro).c_str(),
        device_name_or_empty(mImu.accelerometer).c_str());
    }
  }

  mGps.gps = find_device_by_name("gps", WB_NODE_GPS);
  if (mGps.gps == 0) {
    mGps.gps = find_first_device_by_type(WB_NODE_GPS);
  }

  if (mGps.gps != 0) {
    wb_gps_enable(mGps.gps, wb_robot_get_basic_time_step());
    const char * robot_name = wb_robot_get_name();
    const std::string robot_topic_prefix =
      (robot_name != nullptr && robot_name[0] != '\0') ? robot_name : "webots_robot";
    mGps.speed_vector_topic = robot_topic_prefix + "/gps/speed_vector";
    mGps.speed_vector_publisher = mNode->create_publisher<geometry_msgs::msg::Vector3>(
      mGps.speed_vector_topic, rclcpp::SensorDataQoS());

    RCLCPP_INFO(
      rclcpp::get_logger("webots_bridge"),
      "Publishing Webots GPS speed vector from device '%s' on topic '%s'",
      device_name_or_empty(mGps.gps).c_str(), mGps.speed_vector_topic.c_str());
  } else {
    RCLCPP_WARN(
      rclcpp::get_logger("webots_bridge"),
      "No Webots GPS device found, topic '<robot_name>/gps/speed_vector' will not be published");
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn WebotsBridge::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS) {
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::ERROR;
  }
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> WebotsBridge::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (Joint & joint : mJoints) {
    if (joint.sensor) {
      interfaces.emplace_back(hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_POSITION, &(joint.position)));
      interfaces.emplace_back(hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_VELOCITY, &(joint.velocity)));
      interfaces.emplace_back(hardware_interface::StateInterface(
        joint.name, hardware_interface::HW_IF_EFFORT, &(joint.effort)));
    }
  }

  for (hardware_interface::ComponentInfo component : info_.sensors) {
    if (component.name == mImu.name) {
      for (uint i = 0; i < 4; i++) {
        interfaces.emplace_back(hardware_interface::StateInterface(
          component.name, component.state_interfaces[i].name, &mImu.orientation[i]));
      }
      for (uint i = 0; i < 3; i++) {
        interfaces.emplace_back(hardware_interface::StateInterface(
          component.name, component.state_interfaces[i + 4].name, &mImu.angular_velocity[i]));
      }
      for (uint i = 0; i < 3; i++) {
        interfaces.emplace_back(hardware_interface::StateInterface(
          component.name, component.state_interfaces[i + 7].name, &mImu.linear_acceleration[i]));
      }
    }
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface> WebotsBridge::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (Joint & joint : mJoints) {
    if (joint.motor) {
      interfaces.emplace_back(hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_POSITION, &(joint.positionCommand)));

      interfaces.emplace_back(hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_EFFORT, &(joint.effortCommand)));

      interfaces.emplace_back(hardware_interface::CommandInterface(
        joint.name, hardware_interface::HW_IF_VELOCITY, &(joint.velocityCommand)));

      interfaces.emplace_back(hardware_interface::CommandInterface(joint.name, "kp", &(joint.kp)));

      interfaces.emplace_back(hardware_interface::CommandInterface(joint.name, "kd", &(joint.kd)));
    }
  }
  return interfaces;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn WebotsBridge::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
WebotsBridge::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type WebotsBridge::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  static double lastReadTime = 0;

  const double deltaTime = wb_robot_get_time() - lastReadTime;
  lastReadTime = wb_robot_get_time();

  for (Joint & joint : mJoints) {
    if (joint.sensor) {
      const double position = wb_position_sensor_get_value(joint.sensor);
      const double velocity =
        std::isnan(joint.position) ? NAN : (position - joint.position) / deltaTime;

      if (!std::isnan(joint.velocity)) {
        joint.acceleration = (joint.velocity - velocity) / deltaTime;
      }
      joint.velocity = velocity;
      joint.position = position;
      auto effort = joint.kp * (joint.positionCommand - joint.position) +
                    joint.kd * (joint.velocityCommand - joint.velocity) + joint.effortCommand;
      joint.effort = wb_motor_get_torque_feedback(joint.motor) + effort;
    }
  }

  if (mImu.accelerometer != 0) {
    const double * values = wb_accelerometer_get_values(mImu.accelerometer);
    mImu.linear_acceleration[0] = values[0];
    mImu.linear_acceleration[1] = values[1];
    mImu.linear_acceleration[2] = values[2];
  }

  if (mImu.gyro != 0) {
    const double * values = wb_gyro_get_values(mImu.gyro);
    mImu.angular_velocity[0] = values[0];
    mImu.angular_velocity[1] = values[1];
    mImu.angular_velocity[2] = values[2];
  }

  if (mImu.inertialUnit != 0) {
    const double * values = wb_inertial_unit_get_quaternion(mImu.inertialUnit);
    mImu.orientation[0] = values[0];
    mImu.orientation[1] = values[1];
    mImu.orientation[2] = values[2];
    mImu.orientation[3] = values[3];
  }

  if (mGps.gps != 0 && mGps.speed_vector_publisher != nullptr) {
    const double * values = wb_gps_get_speed_vector(mGps.gps);
    mGps.speed_vector_msg.x = values[0];
    mGps.speed_vector_msg.y = values[1];
    mGps.speed_vector_msg.z = values[2];
    mGps.speed_vector_publisher->publish(mGps.speed_vector_msg);
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type WebotsBridge::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  std::vector<std::string> wheel_debug_entries;
  std::vector<std::string> leg_debug_entries;
  for (Joint & joint : mJoints) {
    if (joint.motor) {
      const auto raw_effort =
        joint.kp * (joint.positionCommand - joint.position) +
        joint.kd * (joint.velocityCommand - joint.velocity) + joint.effortCommand;
      const auto effort = apply_velocity_limit(joint, raw_effort);
      wb_motor_set_torque(joint.motor, effort);

      std::ostringstream entry;
      entry << "name=" << joint.name << " q=" << joint.position << " dq=" << joint.velocity
            << " q_cmd=" << joint.positionCommand << " dq_cmd=" << joint.velocityCommand
            << " kp=" << joint.kp << " kd=" << joint.kd << " tau_ff=" << joint.effortCommand
            << " raw_effort=" << raw_effort << " applied_effort=" << effort;
      if (std::isfinite(joint.velocityLimit)) {
        entry << " vel_limit=" << joint.velocityLimit
              << " limit_active=" << ((std::abs(joint.velocity) >= joint.velocityLimit) ? 1 : 0)
              << " limit_zeroed=" << ((raw_effort != 0.0 && effort == 0.0) ? 1 : 0);
      }
      if (is_wheel_joint_name(joint.name)) {
        wheel_debug_entries.push_back(entry.str());
      } else {
        leg_debug_entries.push_back(entry.str());
      }
    }
  }
  const double now = wb_robot_get_time();
  if (!wheel_debug_entries.empty() && now - lastWheelDebugTime_ > 0.5) {
    lastWheelDebugTime_ = now;
    std::cout << "[WebotsBridge][wheel_applied]";
    for (const auto & entry : wheel_debug_entries) {
      std::cout << " {" << entry << "}";
    }
    std::cout << std::endl;
  }
  if (!leg_debug_entries.empty() && now - lastLegDebugTime_ > 0.5) {
    lastLegDebugTime_ = now;
    std::cout << "[WebotsBridge][leg_applied]";
    for (const auto & entry : leg_debug_entries) {
      std::cout << " {" << entry << "}";
    }
    std::cout << std::endl;
  }
  return hardware_interface::return_type::OK;
}

}  // namespace tita_webots_ros2_control

PLUGINLIB_EXPORT_CLASS(
  tita_webots_ros2_control::WebotsBridge, webots_ros2_control::Ros2ControlSystemInterface)
