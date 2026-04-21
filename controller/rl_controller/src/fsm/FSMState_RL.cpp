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

#include "rl_controller/fsm/FSMState_RL.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef USE_ENGINE
#include "rl_controller/inferrer/engine_inferrer.hpp"
#else
#include "rl_controller/inferrer/onnx_inferrer.hpp"
#endif

FSMState_RL::FSMState_RL(
  std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName)
: FSMState(data, stateName)
{
  rl_params_ = rl_params;
  printf("Get policy: %s in \"%s\" fsm\n", rl_params_->policy_path.c_str(), stateName.c_str());
#ifdef USE_ENGINE
  // printf("Use TensorRT engine\n");
  inferrer_ = std::make_unique<EngineInferrer>();
#else
  // printf("Use ONNX runtime\n");
  inferrer_ = std::make_unique<ONNXInferrer>();
#endif
  // std::filesystem::path policy_path(rl_params_->policy_path);
  // std::string ext = policy_path.extension().string();  // 获取扩展名（含点，如 ".onnx"）
  // if (ext == ".onnx") {
  //   inferrer_ = std::make_unique<ONNXInferrer>();
  // } else if (ext == ".engine") {
  //   inferrer_ = std::make_unique<EngineInferrer>();
  // } else {
  //   throw std::runtime_error("Unsupported file type");
  // }
  inferrer_->loadModel(rl_params_->policy_path);
  inferrer_->setOutput(rl_params_->output_name, rl_params_->num_actions);

  if (
    rl_params_->observations_history_mode != "frame" &&
    rl_params_->observations_history_mode != "term") {
    throw std::runtime_error(
      "[FSMState_RL] Unsupported observations_history_mode: " +
      rl_params_->observations_history_mode);
  }

  if (!rl_params_->observations_dims.empty() &&
      rl_params_->observations_dims.size() != rl_params_->observations_name.size()) {
    throw std::runtime_error(
      "[FSMState_RL] observations_dims size does not match observations_name size");
  }

  obs_term_dims_.resize(rl_params_->observations_name.size());
  int resolved_num_obs = 0;
  for (size_t i = 0; i < rl_params_->observations_name.size(); ++i) {
    long int inferred_dim = infer_observation_dim(rl_params_->observations_name[i]);
    long int configured_dim = inferred_dim;
    if (!rl_params_->observations_dims.empty()) {
      configured_dim = rl_params_->observations_dims[i];
      if (configured_dim != inferred_dim) {
        throw std::runtime_error(
          "[FSMState_RL] observations_dims[" + std::to_string(i) + "] for \"" +
          rl_params_->observations_name[i] + "\" should be " + std::to_string(inferred_dim) +
          ", but got " + std::to_string(configured_dim));
      }
    }
    obs_term_dims_[i] = configured_dim;
    resolved_num_obs += static_cast<int>(configured_dim);
  }

  if (rl_params_->num_obs != resolved_num_obs) {
    throw std::runtime_error(
      "[FSMState_RL] num_obs does not match summed observations_dims: " +
      std::to_string(rl_params_->num_obs) + " vs " + std::to_string(resolved_num_obs));
  }

  obs_vec_.setZero(rl_params_->num_obs);
  obs_history_vec_.setZero(rl_params_->num_obs * rl_params_->history_len);
  action_vec_.setZero(rl_params_->num_actions);
  obs_terms_.resize(rl_params_->observations_name.size());
  obs_term_history_vecs_.resize(rl_params_->observations_name.size());
  obs_.commands.setZero(rl_params_->commands_name.size());
  obs_.dof_pos.setZero(rl_params_->num_actions);
  obs_.dof_vel.setZero(rl_params_->num_actions);
  obs_.last_actions.setZero(rl_params_->num_actions);
  if (rl_params_->action_scales.size() == 1) {
    scalar_t base_scale = rl_params_->action_scales[0];
    rl_params_->action_scales.resize(rl_params_->num_actions, base_scale);
  }

  has_base_lin_vel_xy_observation_ = requires_base_lin_vel_xy();
  if (has_base_lin_vel_xy_observation_) {
    setup_base_lin_vel_subscription();
  }
}

void FSMState_RL::enter()
{
  obs_.reset();
  update_observations();
  initialize_observation_history();
  threadRunning = true;
  if (thread_first_) {
    forward_thread = std::thread(&FSMState_RL::update_forward, this);
    thread_first_ = false;
  }
  stop_update_ = false;
  iter_ = 0;
}

void FSMState_RL::run()
{
  _data->low_cmd->zero();
  DVec<tensor_element_t> pos = d2f(_data->low_state->q);
  DVec<tensor_element_t> vel = d2f(_data->low_state->dq);
  for (auto i : _data->params->wheel_indices) {
    pos[i] = .0f;
  }

  // compute command
  std::vector<tensor_element_t> torques;
  std::vector<std::string> wheel_debug_entries;
  for (int i = 0; i < rl_params_->num_actions; i++) {
    tensor_element_t action_scaled = action_vec_[i] * rl_params_->action_scales[i];
    // printf("sad%f" , rl_params_->action_scales[i]);
    // tensor_element_t torque = 0.0;
    if (rl_params_->control_type == "P") {
      bool is_wheel_joint =
        std::find(_data->params->wheel_indices.begin(), _data->params->wheel_indices.end(), i) !=
        _data->params->wheel_indices.end();
      tensor_element_t command =
        action_scaled + (tensor_element_t)rl_params_->default_joint_angles[i];
      _data->low_cmd->kp(i) = is_wheel_joint ? 0.0 : rl_params_->joint_kp[i];
      _data->low_cmd->kd(i) = is_wheel_joint ? 0.0 : rl_params_->joint_kd[i];
      _data->low_cmd->qd(i) = is_wheel_joint ? 0.0 : command;
      _data->low_cmd->qd_dot(i) = 0.0;
      _data->low_cmd->tau_cmd(i) =
        is_wheel_joint ? rl_params_->joint_kp[i] * command - rl_params_->joint_kd[i] * vel[i] : 0.0;
      if (is_wheel_joint) {
        const scalar_t torque_limit = i < static_cast<int>(_data->params->torque_limit.size())
                                      ? std::abs(_data->params->torque_limit[i])
                                      : std::numeric_limits<scalar_t>::infinity();
        const scalar_t raw_tau = _data->low_cmd->tau_cmd(i);
        const scalar_t final_tau = std::clamp(raw_tau, -torque_limit, torque_limit);
        std::ostringstream entry;
        entry << "idx=" << i << " action=" << action_vec_[i] << " scaled=" << action_scaled
              << " dq=" << vel[i] << " raw_tau=" << raw_tau << " final_tau=" << final_tau;
        wheel_debug_entries.push_back(entry.str());
      }
    } else if (rl_params_->control_type == "P_V") {
      bool is_wheel_joint =
        std::find(_data->params->wheel_indices.begin(), _data->params->wheel_indices.end(), i) !=
        _data->params->wheel_indices.end();
      tensor_element_t command =
        action_scaled + (tensor_element_t)rl_params_->default_joint_angles[i];
      _data->low_cmd->kp(i) = is_wheel_joint ? 0.0 : rl_params_->joint_kp[i];
      _data->low_cmd->kd(i) = rl_params_->joint_kd[i];
      _data->low_cmd->qd(i) = is_wheel_joint ? 0.0 : command;
      _data->low_cmd->qd_dot(i) = is_wheel_joint ? action_scaled : 0.0;
      _data->low_cmd->tau_cmd(i) = 0.0;
      if (is_wheel_joint) {
        std::ostringstream entry;
        entry << "idx=" << i << " action=" << action_vec_[i] << " scaled=" << action_scaled
              << " dq=" << vel[i] << " qd_dot=" << _data->low_cmd->qd_dot(i);
        wheel_debug_entries.push_back(entry.str());
      }
    } else {
      throw std::runtime_error("[FSMState_RL] Unknown control type");
    }
    // torque *= rl_params_->output_torque_scale;
    // torques.push_back(torque);
  }
  const double now = getTimeSecond();
  if (!wheel_debug_entries.empty() && now - last_wheel_debug_time_ > 0.5) {
    last_wheel_debug_time_ = now;
    std::cout << "[FSMState_RL][wheel_debug]";
    for (const auto & entry : wheel_debug_entries) {
      std::cout << " {" << entry << "}";
    }
    std::cout << std::endl;
  }
  // _data->low_cmd->tau_cmd = f2d(vectorToEigen(torques));
}

void FSMState_RL::exit()
{
  stop_update_ = true;
  // std::cout << "exit RL" << std::endl;
}

std::string FSMState_RL::checkTransition()
{
  this->_nextStateName = this->_stateName;
  auto desiredState = _data->rc_data->fsm_name_;
  // 将 switch 替换为 if-else 结构
  if (desiredState.find("rl_") == 0) {
    try {
      size_t number = std::stoi(desiredState.substr(3));
      if (number < _data->params->rl_policy_names.size()) {
        this->_nextStateName = _data->params->rl_policy_names[number];
      }
    } catch (const std::exception & e) {
      std::cerr << "Invalid number format in " << desiredState << std::endl;
    }
  } else if (desiredState == "transform_down") {
    this->_nextStateName = "transform_down";
    // } else if (desiredState == "idle") {
    //   // normal c
    //   this->_nextStateName = "idle";
  }
  return this->_nextStateName;
}

bool FSMState_RL::transition()
{
  run();
  if (
    rl_params_->episode_length > 1e-4 &&
    getTimeSecond() - obs_.phase_start_time < rl_params_->episode_length) {
    return false;
  }  // For some phase policy, we need to wait for the episode to finish
  // You can add the height decend for 8dof robots to get a smooth landing
  return true;
}

long int FSMState_RL::infer_observation_dim(const std::string & observation_name) const
{
  if (observation_name == "ang_vel" || observation_name == "gravity") {
    return 3;
  }
  if (observation_name == "commands") {
    return static_cast<long int>(rl_params_->commands_name.size());
  }
  if (
    observation_name == "dof_pos" || observation_name == "dof_vel" ||
    observation_name == "last_actions") {
    return rl_params_->num_actions;
  }
  if (observation_name == "dof_pos_nwp") {
    return 6;
  }
  if (observation_name == "base_lin_vel_xy") {
    return 2;
  }
  if (observation_name == "phases") {
    return 6;
  }

  throw std::runtime_error("[FSMState_RL] Unknown observation name: " + observation_name);
}

bool FSMState_RL::use_term_history_layout() const
{
  return rl_params_->observations_history_mode == "term";
}

bool FSMState_RL::requires_base_lin_vel_xy() const
{
  return std::find(
           rl_params_->observations_name.begin(), rl_params_->observations_name.end(),
           "base_lin_vel_xy") != rl_params_->observations_name.end();
}

bool FSMState_RL::should_accept_base_lin_vel_sample(
  double now_sec, double & last_time_sec, int rate_hz)
{
  if (rate_hz <= 0) {
    last_time_sec = now_sec;
    return true;
  }

  const double min_interval = 1.0 / static_cast<double>(rate_hz);
  if (last_time_sec >= 0.0 && now_sec - last_time_sec < min_interval) {
    return false;
  }

  last_time_sec = now_sec;
  return true;
}

void FSMState_RL::setup_base_lin_vel_subscription()
{
  if (!_data->node) {
    throw std::runtime_error(
      "[FSMState_RL] base_lin_vel_xy requires a valid ROS node for subscriptions");
  }

  bool use_sim_time = false;
  _data->node->get_parameter("use_sim_time", use_sim_time);
  use_sim_base_lin_vel_source_ = use_sim_time;

  auto qos = rclcpp::SensorDataQoS();
  if (use_sim_base_lin_vel_source_) {
    if (rl_params_->base_lin_vel_xy_sim_topic.empty()) {
      throw std::runtime_error(
        "[FSMState_RL] base_lin_vel_xy sim_topic is empty while use_sim_time=true");
    }
    base_lin_vel_sim_subscription_ = _data->node->create_subscription<geometry_msgs::msg::Vector3>(
      rl_params_->base_lin_vel_xy_sim_topic, qos,
      std::bind(&FSMState_RL::sim_base_lin_vel_cb, this, std::placeholders::_1));
    RCLCPP_INFO(
      _data->node->get_logger(),
      "[FSMState_RL] base_lin_vel_xy uses sim topic %s at %d Hz",
      rl_params_->base_lin_vel_xy_sim_topic.c_str(), rl_params_->base_lin_vel_xy_sim_rate_hz);
    return;
  }

  if (rl_params_->base_lin_vel_xy_hw_topic.empty()) {
    throw std::runtime_error(
      "[FSMState_RL] base_lin_vel_xy hw_topic is empty while use_sim_time=false");
  }
  base_lin_vel_hw_subscription_ = _data->node->create_subscription<std_msgs::msg::Float64>(
    rl_params_->base_lin_vel_xy_hw_topic, qos,
    std::bind(&FSMState_RL::hw_base_lin_vel_cb, this, std::placeholders::_1));
  RCLCPP_INFO(
    _data->node->get_logger(),
    "[FSMState_RL] base_lin_vel_xy uses hardware Float64 x-velocity topic %s at %d Hz",
    rl_params_->base_lin_vel_xy_hw_topic.c_str(), rl_params_->base_lin_vel_xy_hw_rate_hz);
}

void FSMState_RL::sim_base_lin_vel_cb(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
  const double now_sec = _data->node ? _data->node->now().seconds() : getTimeSecond();
  std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
  if (
    !should_accept_base_lin_vel_sample(
      now_sec, last_base_lin_vel_sim_update_time_, rl_params_->base_lin_vel_xy_sim_rate_hz)) {
    return;
  }

  latest_base_lin_vel_world_ << static_cast<tensor_element_t>(msg->x),
    static_cast<tensor_element_t>(msg->y), static_cast<tensor_element_t>(msg->z);
}

void FSMState_RL::hw_base_lin_vel_cb(const std_msgs::msg::Float64::SharedPtr msg)
{
  const double now_sec = _data->node ? _data->node->now().seconds() : getTimeSecond();
  std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
  if (
    !should_accept_base_lin_vel_sample(
      now_sec, last_base_lin_vel_hw_update_time_, rl_params_->base_lin_vel_xy_hw_rate_hz)) {
    return;
  }

  latest_base_lin_vel_body_ << static_cast<tensor_element_t>(msg->data),
    static_cast<tensor_element_t>(0.0), static_cast<tensor_element_t>(0.0);
}

DVec<tensor_element_t> FSMState_RL::build_observation_term(
  const std::string & observation_name, const DVec<tensor_element_t> & pos,
  const DVec<tensor_element_t> & vel)
{
  if (observation_name == "ang_vel") {
    return obs_.ang_vel * rl_params_->ang_vel_scale;
  }
  if (observation_name == "gravity") {
    return obs_.gravity;
  }
  if (observation_name == "commands") {
    return obs_.commands;
  }
  if (observation_name == "dof_pos") {
    return pos * rl_params_->dof_pos_scale;
  }
  if (observation_name == "dof_pos_nwp") {
    std::vector<int> indices;
    const auto & observation_reindex = rl_params_->observation_reindex;
    for (Eigen::Index i = 0; i < pos.size(); ++i) {
      long int original_index = observation_reindex.empty() ? i : observation_reindex[i];
      bool is_wheel_joint =
        std::find(
          _data->params->wheel_indices.begin(), _data->params->wheel_indices.end(),
          original_index) != _data->params->wheel_indices.end();
      if (!is_wheel_joint) {
        indices.push_back(static_cast<int>(i));
      }
    }
    DVec<tensor_element_t> pos_sliced(indices.size());
    for (size_t j = 0; j < indices.size(); ++j) {
      pos_sliced[j] = pos[indices[j]];
    }
    return pos_sliced * static_cast<tensor_element_t>(rl_params_->dof_pos_scale);
  }
  if (observation_name == "dof_vel") {
    return vel * rl_params_->dof_vel_scale;
  }
  if (observation_name == "last_actions") {
    return obs_.last_actions;
  }
  if (observation_name == "base_lin_vel_xy") {
    DVec<tensor_element_t> base_lin_vel_xy(2);
    base_lin_vel_xy << obs_.lin_vel[0], obs_.lin_vel[1];
    return base_lin_vel_xy * static_cast<tensor_element_t>(rl_params_->lin_vel_scale);
  }
  if (observation_name == "phases") {
    return obs_.phases;
  }

  throw std::runtime_error("[FSMState_RL] Unknown observation name: " + observation_name);
}

void FSMState_RL::flatten_term_history()
{
  int offset = 0;
  for (const auto & term_history : obs_term_history_vecs_) {
    obs_history_vec_.segment(offset, term_history.size()) = term_history;
    offset += term_history.size();
  }
}

void FSMState_RL::initialize_observation_history()
{
  if (!use_term_history_layout()) {
    for (int i = 0; i < rl_params_->history_len; i++) {
      obs_history_vec_.segment(i * rl_params_->num_obs, rl_params_->num_obs) = obs_vec_;
    }
    return;
  }

  for (size_t i = 0; i < obs_terms_.size(); ++i) {
    const long int term_dim = obs_term_dims_[i];
    obs_term_history_vecs_[i].setZero(term_dim * rl_params_->history_len);
    for (int j = 0; j < rl_params_->history_len; ++j) {
      obs_term_history_vecs_[i].segment(j * term_dim, term_dim) = obs_terms_[i];
    }
  }
  flatten_term_history();
}

void FSMState_RL::append_observation_history()
{
  if (!use_term_history_layout()) {
    obs_history_vec_.head(obs_history_vec_.size() - obs_vec_.size()) =
      obs_history_vec_.tail(obs_history_vec_.size() - obs_vec_.size());
    obs_history_vec_.tail(obs_vec_.size()) = obs_vec_;
    return;
  }

  for (size_t i = 0; i < obs_terms_.size(); ++i) {
    auto & term_history = obs_term_history_vecs_[i];
    const long int term_dim = obs_term_dims_[i];
    term_history.head(term_history.size() - term_dim) = term_history.tail(term_history.size() - term_dim);
    term_history.tail(term_dim) = obs_terms_[i];
  }
  flatten_term_history();
}

void FSMState_RL::update_observations()
{
  obs_.dof_pos = d2f(_data->low_state->q);
  obs_.dof_vel = d2f(_data->low_state->dq);
  for (auto i : _data->params->wheel_indices) {
    obs_.dof_pos[i] = .0f;
  }

  obs_.ang_vel = 0.97 * d2f(this->_data->low_state->gyro) + 0.03 * obs_.ang_vel;

  // compute gravity
  auto rBody = d2f(ori::quaternionToRotationMatrix(_data->low_state->quat));
  obs_.gravity = rBody * Vec3<tensor_element_t>(0.0, 0.0, -1.0);
  if (has_base_lin_vel_xy_observation_) {
    std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
    obs_.lin_vel = use_sim_base_lin_vel_source_ ? rBody * latest_base_lin_vel_world_
                                                : latest_base_lin_vel_body_;
  }

  // command
  for (size_t i = 0; i < rl_params_->commands_name.size(); i++) {
    scalar_t command;
    if (rl_params_->commands_name[i] == "lin_vel_x") {
      command = rl_params_->commands_gain[i] * _data->rc_data->twist_linear[point::X] +
                rl_params_->commands_comp[i];
    } else if (rl_params_->commands_name[i] == "lin_vel_y") {
      command = rl_params_->commands_gain[i] * _data->rc_data->twist_linear[point::Y] +
                rl_params_->commands_comp[i];
    } else if (rl_params_->commands_name[i] == "ang_vel_z") {
      command = rl_params_->commands_gain[i] * _data->rc_data->twist_angular[point::Z] +
                rl_params_->commands_comp[i];
    } else if (rl_params_->commands_name[i] == "base_height") {
      command = rl_params_->commands_gain[i] * _data->rc_data->pose_position[point::Z] +
                rl_params_->commands_comp[i];
    } else {
      throw std::runtime_error("Unknown command name: " + rl_params_->commands_name[i]);
    }
    command = std::max(std::min(command, rl_params_->max_commands[i]), rl_params_->min_commands[i]);
    obs_.commands[i] = command * rl_params_->commands_scale[i];
  }
  // phase
  double phase = (getTimeSecond() - obs_.phase_start_time) * M_PI / 2.0f;
  if (getTimeSecond() - obs_.phase_start_time > rl_params_->episode_length) {
    phase = rl_params_->episode_length * M_PI / 2.0f;
  }
  // clang-format off
  obs_.phases << std::sin(phase),
                 std::cos(phase),
                 std::sin(phase/2.0),
                 std::cos(phase/2.0),
                 std::sin(phase/4.0),
                 std::cos(phase/4.0);
  // clang-format on
  // obs_buf
  DVec<tensor_element_t> pos = obs_.dof_pos - d2f(vectorToEigen(rl_params_->default_joint_angles));
  DVec<tensor_element_t> vel = obs_.dof_vel;
  pos = reindex_observation(pos);
  pos = re_sign_observation(pos);
  vel = reindex_observation(vel);
  vel = re_sign_observation(vel);
  int obs_size = 0;
  for (size_t i = 0; i < rl_params_->observations_name.size(); i++) {
    obs_terms_[i] = build_observation_term(rl_params_->observations_name[i], pos, vel);
    if (obs_terms_[i].size() != obs_term_dims_[i]) {
      throw std::runtime_error(
        "[FSMState_RL] Observation term \"" + rl_params_->observations_name[i] +
        "\" size does not match observations_dims");
    }
    obs_size += obs_terms_[i].size();
  }
  if (obs_vec_.size() != obs_size) {
    throw std::runtime_error("Obs size does not correct");
  }
  int offset = 0;
  for (const auto & obs : obs_terms_) {
    obs_vec_.segment(offset, obs.size()) = obs;
    offset += obs.size();
  }
  // // clang-format off
  // obs_vec_ << obs_.ang_vel * rl_params_->ang_vel_scale,
  //             obs_.gravity,
  //             obs_.commands,
  //             pos * rl_params_->dof_pos_scale,
  //             vel * rl_params_->dof_vel_scale,
  //             obs_.last_actions;
  // // clang-format on
}

void FSMState_RL::update_forward()
{
  const long long interval = static_cast<long long>(rl_params_->time_interval * 1000000);
  while (threadRunning) {
    long long _start_time = getSystemTime();

    if (!stop_update_) {
      update_observations();
      std::vector<std::vector<tensor_element_t>> input_datas;
      std::vector<tensor_element_t> input_data_1 = eigenToVector(obs_vec_);
      std::vector<tensor_element_t> input_data_2 = eigenToVector(obs_history_vec_);
      input_datas.push_back(input_data_1);
      input_datas.push_back(input_data_2);
      action_vec_ = vectorToEigen(inferrer_->computeActions(input_datas));
      obs_.last_actions = action_vec_;
      action_vec_ = reindex_action(action_vec_);
      action_vec_ = re_sign_action(action_vec_);
      append_observation_history();
    }
    absoluteWait(_start_time, interval);
  }
  threadRunning = false;
}
