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
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#ifdef USE_ENGINE
#include "rl_controller/inferrer/engine_inferrer.hpp"
#else
#include "rl_controller/inferrer/onnx_inferrer.hpp"
#endif

namespace
{
std::string vec_to_string(const DVec<tensor_element_t> & vec)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "[";
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << vec[i];
  }
  oss << "]";
  return oss.str();
}

std::string vec_to_csv_cell(const DVec<tensor_element_t> & vec)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      oss << ';';
    }
    oss << vec[i];
  }
  return oss.str();
}

std::string vec_to_csv_cell(const Vec3<tensor_element_t> & vec)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << vec[0] << ';' << vec[1] << ';' << vec[2];
  return oss.str();
}

std::unique_ptr<InferrerBase> make_inferrer()
{
#ifdef USE_ENGINE
  return std::make_unique<EngineInferrer>();
#else
  return std::make_unique<ONNXInferrer>();
#endif
}

size_t tensor_element_count(
  const std::vector<int64_t> & shape, const std::string & model_label, size_t input_index)
{
  size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error(
        "[FSMState_RL] " + model_label + " input " + std::to_string(input_index) +
        " has unsupported dynamic/invalid dimension: " + std::to_string(dim));
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}
}  // namespace

FSMState_RL::FSMState_RL(
  std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName)
: FSMState(data, stateName)
{
  rl_params_ = rl_params;
  printf("Get policy: %s in \"%s\" fsm\n", rl_params_->policy_path.c_str(), stateName.c_str());
  inferrer_ = make_inferrer();
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

  has_base_lin_vel_xy_observation_ = requires_base_lin_vel_xy();
  validate_velocity_estimator_config();
  setup_velocity_estimator();
  validate_model_inputs();

  obs_vec_.setZero(rl_params_->num_obs);
  obs_history_vec_.setZero(rl_params_->num_obs * rl_params_->history_len);
  raw_action_vec_.setZero(rl_params_->num_actions);
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

  if (has_base_lin_vel_xy_observation_ && !rl_params_->use_velocity_estimator) {
    setup_base_lin_vel_subscription();
  } else if (has_base_lin_vel_xy_observation_ && rl_params_->use_velocity_estimator) {
    std::cout << "[FSMState_RL] base_lin_vel_xy is supplied by "
              << rl_params_->estimator_policy_path << "; external velocity topic disabled"
              << std::endl;
  }
}

void FSMState_RL::enter()
{
  obs_.reset();
  update_observations();
  initialize_observation_history();
  open_strict_start_log();
  open_hardware_frame_log();
  threadRunning = true;
  stop_update_ = false;
  if (thread_first_) {
    forward_thread = std::thread(&FSMState_RL::update_forward, this);
    thread_first_ = false;
  }
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
  DVec<tensor_element_t> raw_action_snapshot;
  DVec<tensor_element_t> action_snapshot;
  {
    std::lock_guard<std::mutex> lock(action_mutex_);
    raw_action_snapshot = raw_action_vec_;
    action_snapshot = action_vec_;
  }
  std::vector<tensor_element_t> torques;
  std::vector<std::string> wheel_debug_entries;
  std::vector<std::string> leg_debug_entries;
  auto action_source_index = [this](int command_index) -> int {
    if (rl_params_->reindex.empty()) {
      return command_index;
    }
    if (command_index >= 0 && command_index < static_cast<int>(rl_params_->reindex.size())) {
      return static_cast<int>(rl_params_->reindex[command_index]);
    }
    return -1;
  };
  auto action_sign = [this](int command_index) -> scalar_t {
    if (rl_params_->re_sign.empty()) {
      return 1.0;
    }
    if (command_index >= 0 && command_index < static_cast<int>(rl_params_->re_sign.size())) {
      return rl_params_->re_sign[command_index];
    }
    return 1.0;
  };
  auto raw_action_at = [&raw_action_snapshot](int raw_index) -> tensor_element_t {
    if (raw_index >= 0 && raw_index < raw_action_snapshot.size()) {
      return raw_action_snapshot[raw_index];
    }
    return std::numeric_limits<tensor_element_t>::quiet_NaN();
  };
  for (int i = 0; i < rl_params_->num_actions; i++) {
    tensor_element_t action_scaled = action_snapshot[i] * rl_params_->action_scales[i];
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
        const int raw_index = action_source_index(i);
        const scalar_t torque_limit = i < static_cast<int>(_data->params->torque_limit.size())
                                      ? std::abs(_data->params->torque_limit[i])
                                      : std::numeric_limits<scalar_t>::infinity();
        const scalar_t raw_tau = _data->low_cmd->tau_cmd(i);
        const scalar_t final_tau = std::clamp(raw_tau, -torque_limit, torque_limit);
        const scalar_t bridge_torque_if_pvt_false =
          final_tau +
          _data->low_cmd->kp(i) * (_data->low_cmd->qd(i) - _data->low_state->q(i)) +
          _data->low_cmd->kd(i) * (_data->low_cmd->qd_dot(i) - _data->low_state->dq(i));
        std::ostringstream entry;
        entry << "idx=" << i << " raw_idx=" << raw_index << " raw_action=" << raw_action_at(raw_index)
              << " sign=" << action_sign(i) << " mapped_action=" << action_snapshot[i]
              << " action_scale=" << rl_params_->action_scales[i] << " scaled=" << action_scaled
              << " default_q=" << rl_params_->default_joint_angles[i]
              << " command=" << command << " q=" << _data->low_state->q(i) << " dq=" << vel[i]
              << " policy_kp=" << rl_params_->joint_kp[i]
              << " policy_kd=" << rl_params_->joint_kd[i]
              << " low_qd=" << _data->low_cmd->qd(i)
              << " low_qd_dot=" << _data->low_cmd->qd_dot(i)
              << " low_kp=" << _data->low_cmd->kp(i)
              << " low_kd=" << _data->low_cmd->kd(i)
              << " low_tau_pre_clamp=" << _data->low_cmd->tau_cmd(i)
              << " raw_tau=" << raw_tau << " torque_limit=" << torque_limit
              << " final_tau=" << final_tau
              << " bridge_tau_pvt_false_after_clamp=" << bridge_torque_if_pvt_false;
        wheel_debug_entries.push_back(entry.str());
      } else {
        const int raw_index = action_source_index(i);
        const scalar_t pos_err = _data->low_cmd->qd(i) - _data->low_state->q(i);
        const scalar_t vel_err = _data->low_cmd->qd_dot(i) - _data->low_state->dq(i);
        const scalar_t bridge_torque_if_pvt_false =
          _data->low_cmd->tau_cmd(i) +
          _data->low_cmd->kp(i) * pos_err +
          _data->low_cmd->kd(i) * vel_err;
        std::ostringstream entry;
        entry << "idx=" << i << " raw_idx=" << raw_index << " raw_action=" << raw_action_at(raw_index)
              << " sign=" << action_sign(i) << " mapped_action=" << action_snapshot[i]
              << " action_scale=" << rl_params_->action_scales[i] << " scaled=" << action_scaled
              << " default_q=" << rl_params_->default_joint_angles[i]
              << " command=" << command << " q=" << _data->low_state->q(i) << " dq=" << vel[i]
              << " pos_err=" << pos_err << " vel_err=" << vel_err
              << " policy_kp=" << rl_params_->joint_kp[i]
              << " policy_kd=" << rl_params_->joint_kd[i]
              << " low_qd=" << _data->low_cmd->qd(i)
              << " low_qd_dot=" << _data->low_cmd->qd_dot(i)
              << " low_kp=" << _data->low_cmd->kp(i)
              << " low_kd=" << _data->low_cmd->kd(i)
              << " low_tau=" << _data->low_cmd->tau_cmd(i)
              << " bridge_tau_pvt_false=" << bridge_torque_if_pvt_false;
        leg_debug_entries.push_back(entry.str());
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
        const int raw_index = action_source_index(i);
        const scalar_t bridge_torque_if_pvt_false =
          _data->low_cmd->tau_cmd(i) +
          _data->low_cmd->kp(i) * (_data->low_cmd->qd(i) - _data->low_state->q(i)) +
          _data->low_cmd->kd(i) * (_data->low_cmd->qd_dot(i) - _data->low_state->dq(i));
        std::ostringstream entry;
        entry << "idx=" << i << " raw_idx=" << raw_index
              << " raw_action=" << raw_action_at(raw_index) << " sign=" << action_sign(i)
              << " mapped_action=" << action_snapshot[i]
              << " action_scale=" << rl_params_->action_scales[i] << " scaled=" << action_scaled
              << " default_q=" << rl_params_->default_joint_angles[i]
              << " command=" << command << " q=" << _data->low_state->q(i) << " dq=" << vel[i]
              << " policy_kp=" << rl_params_->joint_kp[i]
              << " policy_kd=" << rl_params_->joint_kd[i]
              << " low_qd=" << _data->low_cmd->qd(i)
              << " low_qd_dot=" << _data->low_cmd->qd_dot(i)
              << " low_kp=" << _data->low_cmd->kp(i)
              << " low_kd=" << _data->low_cmd->kd(i)
              << " low_tau_pre_clamp=" << _data->low_cmd->tau_cmd(i)
              << " bridge_tau_pvt_false=" << bridge_torque_if_pvt_false;
        wheel_debug_entries.push_back(entry.str());
      } else {
        const int raw_index = action_source_index(i);
        const scalar_t pos_err = _data->low_cmd->qd(i) - _data->low_state->q(i);
        const scalar_t vel_err = _data->low_cmd->qd_dot(i) - _data->low_state->dq(i);
        const scalar_t bridge_torque_if_pvt_false =
          _data->low_cmd->tau_cmd(i) +
          _data->low_cmd->kp(i) * pos_err +
          _data->low_cmd->kd(i) * vel_err;
        std::ostringstream entry;
        entry << "idx=" << i << " raw_idx=" << raw_index << " raw_action=" << raw_action_at(raw_index)
              << " sign=" << action_sign(i) << " mapped_action=" << action_snapshot[i]
              << " action_scale=" << rl_params_->action_scales[i] << " scaled=" << action_scaled
              << " default_q=" << rl_params_->default_joint_angles[i]
              << " command=" << command << " q=" << _data->low_state->q(i) << " dq=" << vel[i]
              << " pos_err=" << pos_err << " vel_err=" << vel_err
              << " policy_kp=" << rl_params_->joint_kp[i]
              << " policy_kd=" << rl_params_->joint_kd[i]
              << " low_qd=" << _data->low_cmd->qd(i)
              << " low_qd_dot=" << _data->low_cmd->qd_dot(i)
              << " low_kp=" << _data->low_cmd->kp(i)
              << " low_kd=" << _data->low_cmd->kd(i)
              << " low_tau=" << _data->low_cmd->tau_cmd(i)
              << " bridge_tau_pvt_false=" << bridge_torque_if_pvt_false;
        leg_debug_entries.push_back(entry.str());
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
  if (!leg_debug_entries.empty() && now - last_leg_debug_time_ > 0.5) {
    last_leg_debug_time_ = now;
    std::cout << "[FSMState_RL][leg_debug]";
    for (const auto & entry : leg_debug_entries) {
      std::cout << " {" << entry << "}";
    }
    std::cout << std::endl;
  }
  // _data->low_cmd->tau_cmd = f2d(vectorToEigen(torques));
}

void FSMState_RL::exit()
{
  stop_update_ = true;
  close_strict_start_log();
  close_hardware_frame_log();
  // std::cout << "exit RL" << std::endl;
}

void FSMState_RL::open_strict_start_log()
{
  std::lock_guard<std::mutex> lock(strict_log_mutex_);
  strict_policy_step_ = 0;
  if (strict_start_log_.is_open()) {
    strict_start_log_.close();
  }

  strict_start_log_path_ = "/tmp/fsmstate_rl_strict_policy_start_" + _stateName + ".csv";
  strict_start_log_.open(strict_start_log_path_, std::ios::out | std::ios::trunc);
  if (!strict_start_log_.is_open()) {
    std::cerr << "[FSMState_RL][strict_log] failed to open " << strict_start_log_path_
              << std::endl;
    return;
  }

  strict_start_log_
    << "policy_step,time_sec,state,joint_idx,raw_idx,raw_action,mapped_action,action_scale,"
       "scaled,default_q,command,q,dq,control_type,policy_kp,policy_kd,low_qd,low_qd_dot,"
       "low_kp,low_kd,low_tau_pre_clamp,torque_limit,fsm_final_tau,"
       "bridge_tau_pvt_false,raw_actions,mapped_actions,obs_ang_vel,obs_gravity,obs_commands,"
       "obs_lin_vel,obs_vec\n";
  strict_start_log_.flush();
  std::cout << "[FSMState_RL][strict_log] writing first " << kStrictPolicyLogLimit
            << " policy outputs to " << strict_start_log_path_ << std::endl;
}

void FSMState_RL::open_hardware_frame_log()
{
  std::lock_guard<std::mutex> lock(hardware_log_mutex_);
  hardware_frame_step_ = 0;
  if (hardware_frame_log_.is_open()) {
    hardware_frame_log_.close();
  }

  if (!is_hardware_runtime()) {
    hardware_frame_log_path_.clear();
    return;
  }

  hardware_frame_log_path_ = "/tmp/fsmstate_rl_hw_obs_action_" + _stateName + ".csv";
  hardware_frame_log_.open(hardware_frame_log_path_, std::ios::out | std::ios::trunc);
  if (!hardware_frame_log_.is_open()) {
    std::cerr << "[FSMState_RL][hw_frame_log] failed to open " << hardware_frame_log_path_
              << std::endl;
    return;
  }

  hardware_frame_log_ << "frame_step,time_sec,runtime,state,raw_q,raw_dq,raw_pos_rel,policy_pos_rel,"
                         "policy_dq";
  for (const auto & observation_name : rl_params_->observations_name) {
    hardware_frame_log_ << ",obs_" << observation_name;
  }
  hardware_frame_log_ << ",obs_vec,obs_history_vec,raw_actions,mapped_actions\n";
  hardware_frame_log_.flush();
  std::cout << "[FSMState_RL][hw_frame_log] writing per-frame obs/actions to "
            << hardware_frame_log_path_ << std::endl;
}

void FSMState_RL::close_strict_start_log()
{
  std::lock_guard<std::mutex> lock(strict_log_mutex_);
  if (strict_start_log_.is_open()) {
    strict_start_log_.flush();
    strict_start_log_.close();
    std::cout << "[FSMState_RL][strict_log] wrote " << strict_policy_step_
              << " policy outputs to " << strict_start_log_path_ << std::endl;
  }
}

void FSMState_RL::close_hardware_frame_log()
{
  std::lock_guard<std::mutex> lock(hardware_log_mutex_);
  if (hardware_frame_log_.is_open()) {
    hardware_frame_log_.flush();
    hardware_frame_log_.close();
    std::cout << "[FSMState_RL][hw_frame_log] wrote " << hardware_frame_step_
              << " frames to " << hardware_frame_log_path_ << std::endl;
  }
}

void FSMState_RL::log_strict_policy_output(
  const DVec<tensor_element_t> & raw_actions,
  const DVec<tensor_element_t> & mapped_actions)
{
  std::lock_guard<std::mutex> lock(strict_log_mutex_);
  if (!strict_start_log_.is_open() || strict_policy_step_ >= kStrictPolicyLogLimit) {
    return;
  }

  const auto raw_action_at = [&raw_actions](int raw_index) -> tensor_element_t {
    if (raw_index >= 0 && raw_index < raw_actions.size()) {
      return raw_actions[raw_index];
    }
    return std::numeric_limits<tensor_element_t>::quiet_NaN();
  };
  const auto action_source_index = [this](int command_index) -> int {
    if (rl_params_->reindex.empty()) {
      return command_index;
    }
    if (command_index >= 0 && command_index < static_cast<int>(rl_params_->reindex.size())) {
      return static_cast<int>(rl_params_->reindex[command_index]);
    }
    return -1;
  };

  DVec<tensor_element_t> q = d2f(_data->low_state->q);
  DVec<tensor_element_t> dq = d2f(_data->low_state->dq);
  const double now_sec = getTimeSecond();

  strict_start_log_ << std::fixed << std::setprecision(9);
  for (const auto wheel_index_long : _data->params->wheel_indices) {
    const int i = static_cast<int>(wheel_index_long);
    if (i < 0 || i >= mapped_actions.size() ||
        i >= static_cast<int>(rl_params_->action_scales.size()) ||
        i >= static_cast<int>(rl_params_->default_joint_angles.size()) ||
        i >= static_cast<int>(rl_params_->joint_kp.size()) ||
        i >= static_cast<int>(rl_params_->joint_kd.size())) {
      continue;
    }

    const int raw_index = action_source_index(i);
    const tensor_element_t action_scaled = mapped_actions[i] * rl_params_->action_scales[i];
    const tensor_element_t command =
      action_scaled + static_cast<tensor_element_t>(rl_params_->default_joint_angles[i]);
    const scalar_t torque_limit = i < static_cast<int>(_data->params->torque_limit.size())
                                    ? std::abs(_data->params->torque_limit[i])
                                    : std::numeric_limits<scalar_t>::infinity();

    scalar_t low_qd = 0.0;
    scalar_t low_qd_dot = 0.0;
    scalar_t low_kp = 0.0;
    scalar_t low_kd = 0.0;
    scalar_t low_tau_pre_clamp = 0.0;
    scalar_t fsm_final_tau = 0.0;
    scalar_t bridge_tau_pvt_false = 0.0;

    if (rl_params_->control_type == "P") {
      low_tau_pre_clamp = rl_params_->joint_kp[i] * command - rl_params_->joint_kd[i] * dq[i];
      fsm_final_tau = std::clamp(low_tau_pre_clamp, -torque_limit, torque_limit);
      bridge_tau_pvt_false = fsm_final_tau;
    } else if (rl_params_->control_type == "P_V") {
      low_qd_dot = action_scaled;
      low_kd = rl_params_->joint_kd[i];
      low_tau_pre_clamp = 0.0;
      fsm_final_tau = std::clamp(low_tau_pre_clamp, -torque_limit, torque_limit);
      bridge_tau_pvt_false = low_kd * (low_qd_dot - dq[i]);
    } else {
      continue;
    }

    strict_start_log_
      << strict_policy_step_ << ',' << now_sec << ',' << _stateName << ',' << i << ','
      << raw_index << ',' << raw_action_at(raw_index) << ',' << mapped_actions[i] << ','
      << rl_params_->action_scales[i] << ',' << action_scaled << ','
      << rl_params_->default_joint_angles[i] << ',' << command << ',' << q[i] << ',' << dq[i]
      << ',' << rl_params_->control_type << ',' << rl_params_->joint_kp[i] << ','
      << rl_params_->joint_kd[i] << ',' << low_qd << ',' << low_qd_dot << ',' << low_kp << ','
      << low_kd << ',' << low_tau_pre_clamp << ',' << torque_limit << ',' << fsm_final_tau
      << ',' << bridge_tau_pvt_false << ",\"" << vec_to_csv_cell(raw_actions) << "\",\""
      << vec_to_csv_cell(mapped_actions) << "\",\"" << vec_to_csv_cell(obs_.ang_vel)
      << "\",\"" << vec_to_csv_cell(obs_.gravity) << "\",\"" << vec_to_csv_cell(obs_.commands)
      << "\",\"" << vec_to_csv_cell(obs_.lin_vel) << "\",\"" << vec_to_csv_cell(obs_vec_)
      << "\"\n";
  }

  strict_start_log_.flush();
  strict_policy_step_++;
  if (strict_policy_step_ == kStrictPolicyLogLimit) {
    std::cout << "[FSMState_RL][strict_log] reached " << kStrictPolicyLogLimit
              << " policy outputs in " << strict_start_log_path_ << std::endl;
  }
}

bool FSMState_RL::is_hardware_runtime() const
{
  if (!_data || !_data->node) {
    return false;
  }

  bool use_sim_time = false;
  _data->node->get_parameter("use_sim_time", use_sim_time);
  return !use_sim_time;
}

std::string FSMState_RL::runtime_label() const
{
  return is_hardware_runtime() ? "hardware" : "sim";
}

std::string FSMState_RL::leg_label(size_t leg_index, size_t leg_count) const
{
  if (leg_count == 2) {
    return leg_index == 0 ? "left_leg" : "right_leg";
  }
  if (leg_count == 4) {
    static const std::array<std::string, 4> kQuadrupedLegLabels = {
      "front_left", "front_right", "rear_left", "rear_right"};
    if (leg_index < kQuadrupedLegLabels.size()) {
      return kQuadrupedLegLabels[leg_index];
    }
  }
  return "leg_" + std::to_string(leg_index);
}

void FSMState_RL::print_latest_frame_debug(
  const DVec<tensor_element_t> & raw_actions,
  const DVec<tensor_element_t> & mapped_actions)
{
  const double now_sec = getTimeSecond();
  if (now_sec - last_obs_debug_time_ <= 0.5) {
    return;
  }
  last_obs_debug_time_ = now_sec;

  const DVec<tensor_element_t> raw_q = d2f(_data->low_state->q);
  const DVec<tensor_element_t> raw_dq = d2f(_data->low_state->dq);
  const DVec<tensor_element_t> default_q = d2f(vectorToEigen(rl_params_->default_joint_angles));
  const DVec<tensor_element_t> raw_pos_rel = raw_q - default_q;

  DVec<tensor_element_t> policy_pos_rel = obs_.dof_pos - default_q;
  DVec<tensor_element_t> policy_dq = obs_.dof_vel;
  policy_pos_rel = reindex_observation(policy_pos_rel);
  policy_pos_rel = re_sign_observation(policy_pos_rel);
  policy_dq = reindex_observation(policy_dq);
  policy_dq = re_sign_observation(policy_dq);

  size_t leg_count = !_data->params->hip_indices.empty() ? _data->params->hip_indices.size() : 0;
  if (leg_count == 0) {
    if (raw_q.size() > 0 && raw_q.size() % 4 == 0) {
      leg_count = static_cast<size_t>(raw_q.size() / 4);
    } else if (raw_q.size() > 0 && raw_q.size() % 3 == 0) {
      leg_count = static_cast<size_t>(raw_q.size() / 3);
    } else {
      leg_count = 1;
    }
  }
  const size_t leg_dof = std::max<size_t>(1, static_cast<size_t>(raw_q.size()) / leg_count);

  std::ostringstream frame_stream;
  frame_stream << "[FSMState_RL][frame_debug]\n";
  frame_stream << std::fixed << std::setprecision(4);
  frame_stream << "  runtime=" << runtime_label() << " state=" << _stateName
               << " time_sec=" << now_sec << '\n';
  for (size_t leg_index = 0; leg_index < leg_count; ++leg_index) {
    const Eigen::Index offset = static_cast<Eigen::Index>(leg_index * leg_dof);
    const Eigen::Index count = std::min<Eigen::Index>(
      static_cast<Eigen::Index>(leg_dof), raw_q.size() - offset);
    if (offset >= raw_q.size() || count <= 0) {
      continue;
    }

    const DVec<tensor_element_t> leg_q = raw_q.segment(offset, count);
    const DVec<tensor_element_t> leg_default = default_q.segment(offset, count);
    const DVec<tensor_element_t> leg_delta = raw_pos_rel.segment(offset, count);
    frame_stream << "  " << leg_label(leg_index, leg_count) << " q=" << vec_to_string(leg_q)
                 << " default=" << vec_to_string(leg_default)
                 << " delta=" << vec_to_string(leg_delta) << '\n';
  }
  for (size_t i = 0; i < obs_terms_.size(); ++i) {
    frame_stream << "  obs." << rl_params_->observations_name[i] << "="
                 << vec_to_string(obs_terms_[i]) << '\n';
  }
  frame_stream << "  obs.obs_vec=" << vec_to_string(obs_vec_) << '\n';
  frame_stream << "  policy.raw_actions=" << vec_to_string(raw_actions) << '\n';
  frame_stream << "  policy.mapped_actions=" << vec_to_string(mapped_actions);
  std::cout << frame_stream.str() << std::endl;
}

void FSMState_RL::log_hardware_frame(
  const DVec<tensor_element_t> & raw_actions,
  const DVec<tensor_element_t> & mapped_actions)
{
  if (!is_hardware_runtime()) {
    return;
  }

  std::lock_guard<std::mutex> lock(hardware_log_mutex_);
  if (!hardware_frame_log_.is_open()) {
    return;
  }

  const DVec<tensor_element_t> raw_q = d2f(_data->low_state->q);
  const DVec<tensor_element_t> raw_dq = d2f(_data->low_state->dq);
  const DVec<tensor_element_t> default_q = d2f(vectorToEigen(rl_params_->default_joint_angles));
  const DVec<tensor_element_t> raw_pos_rel = raw_q - default_q;

  DVec<tensor_element_t> policy_pos_rel = obs_.dof_pos - default_q;
  DVec<tensor_element_t> policy_dq = obs_.dof_vel;
  policy_pos_rel = reindex_observation(policy_pos_rel);
  policy_pos_rel = re_sign_observation(policy_pos_rel);
  policy_dq = reindex_observation(policy_dq);
  policy_dq = re_sign_observation(policy_dq);

  hardware_frame_log_ << std::fixed << std::setprecision(9) << hardware_frame_step_++ << ','
                      << getTimeSecond() << ',' << runtime_label() << ',' << _stateName << ",\""
                      << vec_to_csv_cell(raw_q) << "\",\"" << vec_to_csv_cell(raw_dq) << "\",\""
                      << vec_to_csv_cell(raw_pos_rel) << "\",\""
                      << vec_to_csv_cell(policy_pos_rel) << "\",\""
                      << vec_to_csv_cell(policy_dq) << '"';
  for (const auto & obs_term : obs_terms_) {
    hardware_frame_log_ << ",\"" << vec_to_csv_cell(obs_term) << '"';
  }
  hardware_frame_log_ << ",\"" << vec_to_csv_cell(obs_vec_) << "\",\""
                      << vec_to_csv_cell(obs_history_vec_) << "\",\""
                      << vec_to_csv_cell(raw_actions) << "\",\""
                      << vec_to_csv_cell(mapped_actions) << "\"\n";
  hardware_frame_log_.flush();
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

void FSMState_RL::validate_velocity_estimator_config() const
{
  if (!rl_params_->use_velocity_estimator) {
    return;
  }

#ifdef USE_ENGINE
  throw std::runtime_error(
    "[FSMState_RL] use_velocity_estimator currently supports ONNX runtime builds only; "
    "rebuild rl_controller with USE_ENGINE=OFF");
#endif

  if (!use_term_history_layout()) {
    throw std::runtime_error(
      "[FSMState_RL] use_velocity_estimator requires observations_history_mode=\"term\"");
  }
  if (!has_base_lin_vel_xy_observation_) {
    throw std::runtime_error(
      "[FSMState_RL] use_velocity_estimator requires base_lin_vel_xy in observations_name");
  }
  if (rl_params_->estimator_policy_path.empty()) {
    throw std::runtime_error(
      "[FSMState_RL] estimator_policy_path is required when use_velocity_estimator=true");
  }
  if (rl_params_->estimator_output_name.empty()) {
    throw std::runtime_error(
      "[FSMState_RL] estimator_output_name is required when use_velocity_estimator=true");
  }
  if (rl_params_->estimator_history_len <= 0) {
    throw std::runtime_error(
      "[FSMState_RL] estimator_history_len must be positive when use_velocity_estimator=true");
  }
  if (rl_params_->estimator_history_len > rl_params_->history_len) {
    throw std::runtime_error(
      "[FSMState_RL] estimator_history_len must be <= history_len, estimator=" +
      std::to_string(rl_params_->estimator_history_len) +
      ", history=" + std::to_string(rl_params_->history_len));
  }
  if (std::fabs(static_cast<double>(rl_params_->lin_vel_scale)) < 1e-6) {
    throw std::runtime_error(
      "[FSMState_RL] lin_vel_scale must be non-zero when use_velocity_estimator=true");
  }
}

void FSMState_RL::setup_velocity_estimator()
{
  if (!rl_params_->use_velocity_estimator) {
    return;
  }

  estimator_inferrer_ = make_inferrer();
  estimator_inferrer_->loadModel(rl_params_->estimator_policy_path);
  estimator_inferrer_->setOutput(rl_params_->estimator_output_name, 2);
  std::cout << "[FSMState_RL] velocity estimator enabled: estimator_policy_path="
            << rl_params_->estimator_policy_path << ", actor_policy_path="
            << rl_params_->policy_path << ", estimator_history_len="
            << rl_params_->estimator_history_len << std::endl;
}

void FSMState_RL::validate_model_inputs() const
{
  if (!rl_params_->use_velocity_estimator) {
    return;
  }

  const auto & actor_input_shapes = inferrer_->getInputShapes();
  if (rl_params_->policy_type == "ppo") {
    if (actor_input_shapes.size() != 1) {
      throw std::runtime_error(
        "[FSMState_RL] PPO actor model must have one input, got " +
        std::to_string(actor_input_shapes.size()));
    }
    const size_t actor_input_dim = tensor_element_count(actor_input_shapes[0], "actor", 0);
    if (actor_input_dim != actor_history_input_dim()) {
      throw std::runtime_error(
        "[FSMState_RL] PPO actor input dim mismatch, configured=" +
        std::to_string(actor_history_input_dim()) +
        ", model=" + std::to_string(actor_input_dim));
    }
  } else if (rl_params_->policy_type == "np3o") {
    if (actor_input_shapes.size() != 2) {
      throw std::runtime_error(
        "[FSMState_RL] NP3O actor model must have two inputs, got " +
        std::to_string(actor_input_shapes.size()));
    }
    const size_t actor_obs_dim = tensor_element_count(actor_input_shapes[0], "actor", 0);
    const size_t actor_history_dim = tensor_element_count(actor_input_shapes[1], "actor", 1);
    if (actor_obs_dim != static_cast<size_t>(rl_params_->num_obs)) {
      throw std::runtime_error(
        "[FSMState_RL] NP3O actor obs input dim mismatch, configured=" +
        std::to_string(rl_params_->num_obs) + ", model=" + std::to_string(actor_obs_dim));
    }
    if (actor_history_dim != actor_history_input_dim()) {
      throw std::runtime_error(
        "[FSMState_RL] NP3O actor history input dim mismatch, configured=" +
        std::to_string(actor_history_input_dim()) +
        ", model=" + std::to_string(actor_history_dim));
    }
  }

  if (!estimator_inferrer_) {
    throw std::runtime_error("[FSMState_RL] estimator inferrer was not initialized");
  }
  const auto & estimator_input_shapes = estimator_inferrer_->getInputShapes();
  if (estimator_input_shapes.size() != 1) {
    throw std::runtime_error(
      "[FSMState_RL] estimator model must have one input, got " +
      std::to_string(estimator_input_shapes.size()));
  }
  const size_t estimator_input_dim =
    tensor_element_count(estimator_input_shapes[0], "estimator", 0);
  if (estimator_input_dim != estimator_history_input_dim()) {
    throw std::runtime_error(
      "[FSMState_RL] estimator input dim mismatch, configured=" +
      std::to_string(estimator_history_input_dim()) +
      ", model=" + std::to_string(estimator_input_dim));
  }
}

size_t FSMState_RL::actor_history_input_dim() const
{
  return static_cast<size_t>(rl_params_->num_obs) * static_cast<size_t>(rl_params_->history_len);
}

size_t FSMState_RL::estimator_history_input_dim() const
{
  size_t input_dim = 0;
  for (size_t i = 0; i < rl_params_->observations_name.size(); ++i) {
    if (rl_params_->observations_name[i] == "base_lin_vel_xy") {
      continue;
    }
    input_dim +=
      static_cast<size_t>(obs_term_dims_[i]) * static_cast<size_t>(rl_params_->estimator_history_len);
  }
  return input_dim;
}

std::vector<tensor_element_t> FSMState_RL::build_estimator_history_input() const
{
  if (!use_term_history_layout()) {
    throw std::runtime_error(
      "[FSMState_RL] estimator input assembly requires term history layout");
  }

  std::vector<tensor_element_t> estimator_input;
  estimator_input.reserve(estimator_history_input_dim());
  for (size_t i = 0; i < obs_term_history_vecs_.size(); ++i) {
    if (rl_params_->observations_name[i] == "base_lin_vel_xy") {
      continue;
    }
    const size_t term_dim = static_cast<size_t>(obs_term_dims_[i]);
    const size_t copy_count = term_dim * static_cast<size_t>(rl_params_->estimator_history_len);
    const auto & term_history = obs_term_history_vecs_[i];
    if (static_cast<size_t>(term_history.size()) < copy_count) {
      throw std::runtime_error(
        "[FSMState_RL] estimator history for \"" + rl_params_->observations_name[i] +
        "\" is shorter than estimator_history_len");
    }
    const size_t offset = static_cast<size_t>(term_history.size()) - copy_count;
    for (size_t j = 0; j < copy_count; ++j) {
      estimator_input.push_back(term_history[static_cast<Eigen::Index>(offset + j)]);
    }
  }
  return estimator_input;
}

bool FSMState_RL::run_velocity_estimator()
{
  if (!rl_params_->use_velocity_estimator) {
    return false;
  }
  if (!estimator_inferrer_) {
    throw std::runtime_error("[FSMState_RL] estimator inferrer is not initialized");
  }

  std::vector<std::vector<tensor_element_t>> input_datas;
  input_datas.push_back(build_estimator_history_input());
  const auto estimated_velocity = estimator_inferrer_->computeActions(input_datas);
  if (estimated_velocity.size() != 2) {
    throw std::runtime_error(
      "[FSMState_RL] estimator output size mismatch, expected 2, got " +
      std::to_string(estimated_velocity.size()));
  }

  estimated_base_lin_vel_body_[0] = estimated_velocity[0];
  estimated_base_lin_vel_body_[1] = estimated_velocity[1];
  estimated_base_lin_vel_body_[2] = static_cast<tensor_element_t>(0.0);
  refresh_estimated_base_lin_vel_observation();
  return true;
}

void FSMState_RL::refresh_estimated_base_lin_vel_observation()
{
  if (!rl_params_->use_velocity_estimator || !has_base_lin_vel_xy_observation_) {
    return;
  }

  obs_.lin_vel = estimated_base_lin_vel_body_;
  Eigen::Index offset = 0;
  for (size_t i = 0; i < rl_params_->observations_name.size(); ++i) {
    const Eigen::Index term_dim = static_cast<Eigen::Index>(obs_term_dims_[i]);
    if (rl_params_->observations_name[i] != "base_lin_vel_xy") {
      offset += term_dim;
      continue;
    }

    DVec<tensor_element_t> base_lin_vel_xy(2);
    base_lin_vel_xy << obs_.lin_vel[0], obs_.lin_vel[1];
    obs_terms_[i] = base_lin_vel_xy * static_cast<tensor_element_t>(rl_params_->lin_vel_scale);
    obs_vec_.segment(offset, term_dim) = obs_terms_[i];

    auto & term_history = obs_term_history_vecs_[i];
    if (term_history.size() >= term_dim) {
      term_history.tail(term_dim) = obs_terms_[i];
      flatten_term_history();
    }
    return;
  }
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
    if (rl_params_->use_velocity_estimator) {
      obs_.lin_vel = estimated_base_lin_vel_body_;
    } else {
      std::lock_guard<std::mutex> lock(base_lin_vel_mutex_);
      obs_.lin_vel = use_sim_base_lin_vel_source_ ? rBody * latest_base_lin_vel_world_
                                                  : latest_base_lin_vel_body_;
    }
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
      if (rl_params_->use_velocity_estimator) {
        append_observation_history();
        run_velocity_estimator();
      }
      std::vector<std::vector<tensor_element_t>> input_datas;
      std::vector<tensor_element_t> input_data_1 = eigenToVector(obs_vec_);
      std::vector<tensor_element_t> input_data_2 = eigenToVector(obs_history_vec_);
      input_datas.push_back(input_data_1);
      input_datas.push_back(input_data_2);
      auto raw_actions = vectorToEigen(inferrer_->computeActions(input_datas));
      auto mapped_actions = reindex_action(raw_actions);
      mapped_actions = re_sign_action(mapped_actions);
      log_strict_policy_output(raw_actions, mapped_actions);
      print_latest_frame_debug(raw_actions, mapped_actions);
      log_hardware_frame(raw_actions, mapped_actions);
      obs_.last_actions = raw_actions;
      {
        std::lock_guard<std::mutex> lock(action_mutex_);
        raw_action_vec_ = raw_actions;
        action_vec_ = mapped_actions;
      }
      if (!rl_params_->use_velocity_estimator) {
        append_observation_history();
      }
    }
    absoluteWait(_start_time, interval);
  }
  threadRunning = false;
}
