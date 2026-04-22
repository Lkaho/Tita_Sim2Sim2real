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

#ifndef RL_CONTROLLER__FSM__FSMSTATE_RL_H_
#define RL_CONTROLLER__FSM__FSMSTATE_RL_H_

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "FSMState.h"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rl_controller/common/timeMarker.h"
#include "rl_controller/inferrer/inferrer_base.hpp"
#include "std_msgs/msg/float64.hpp"

struct Observations
{
  Vec3<tensor_element_t> lin_vel;
  Vec3<tensor_element_t> ang_vel;
  Vec3<tensor_element_t> gravity;
  DVec<tensor_element_t> commands;
  DVec<tensor_element_t> dof_pos;
  DVec<tensor_element_t> dof_vel;
  DVec<tensor_element_t> last_actions;
  DVec<tensor_element_t> phases{6};  //
  double phase_start_time;

  void reset()
  {
    lin_vel.setZero();
    ang_vel.setZero();
    gravity.setZero();
    if (
      commands.size() == 0 || dof_pos.size() == 0 || dof_vel.size() == 0 ||
      last_actions.size() == 0) {
      throw std::runtime_error("Observations vectors must be initialized before reset");
    } else {
      commands.setZero();
      dof_pos.setZero();
      dof_vel.setZero();
      last_actions.setZero();
    }
    phase_start_time = getTimeSecond();
  }
};
class FSMState_RL : public FSMState
{
public:
  FSMState_RL(
    std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName);
  virtual ~FSMState_RL() {}

  // Behavior to be carried out when entering a state
  void enter();

  // Run the normal behavior for the state
  void run();

  // Checks for any transition triggers
  std::string checkTransition();

  bool transition();

  // Manages state specific transitions
  //   TransitionData transition();

  // Behavior to be carried out when exiting a state
  void exit();

  //   TransitionData testTransition();

protected:
  virtual void update_observations();
  virtual void update_forward();
  DVec<tensor_element_t> build_observation_term(
    const std::string & observation_name, const DVec<tensor_element_t> & pos,
    const DVec<tensor_element_t> & vel);
  void initialize_observation_history();
  void append_observation_history();
  void flatten_term_history();
  long int infer_observation_dim(const std::string & observation_name) const;
  bool use_term_history_layout() const;
  bool requires_base_lin_vel_xy() const;
  void setup_base_lin_vel_subscription();
  void sim_base_lin_vel_cb(const geometry_msgs::msg::Vector3::SharedPtr msg);
  void hw_base_lin_vel_cb(const std_msgs::msg::Float64::SharedPtr msg);
  bool should_accept_base_lin_vel_sample(double now_sec, double & last_time_sec, int rate_hz);
  void open_strict_start_log();
  void close_strict_start_log();
  void log_strict_policy_output(
    const DVec<tensor_element_t> & raw_actions,
    const DVec<tensor_element_t> & mapped_actions);

  DVec<tensor_element_t> apply_reindex(
    const DVec<tensor_element_t> & vec, const std::vector<long int> & reindex_map) const
  {
    if (reindex_map.empty()) {
      return vec;
    } else {
      DVec<tensor_element_t> vec_reindex = vec;
      for (auto i = 0UL; i < reindex_map.size(); i++) {
        vec_reindex[i] = vec[reindex_map[i]];
      }
      return vec_reindex;
    }
  };

  DVec<tensor_element_t> apply_re_sign(
    const DVec<tensor_element_t> & vec, const std::vector<scalar_t> & sign_map) const
  {
    if (sign_map.empty()) {
      return vec;
    } else {
      DVec<tensor_element_t> vec_re_sign = vec;
      for (auto i = 0UL; i < sign_map.size(); i++) {
        vec_re_sign[i] = sign_map[i] * vec[i];
      }
      return vec_re_sign;
    }
  };

  DVec<tensor_element_t> reindex_action(const DVec<tensor_element_t> & vec) const
  {
    return apply_reindex(vec, rl_params_->reindex);
  };

  DVec<tensor_element_t> re_sign_action(const DVec<tensor_element_t> & vec) const
  {
    return apply_re_sign(vec, rl_params_->re_sign);
  };

  DVec<tensor_element_t> reindex_observation(const DVec<tensor_element_t> & vec) const
  {
    return apply_reindex(vec, rl_params_->observation_reindex);
  };

  DVec<tensor_element_t> re_sign_observation(const DVec<tensor_element_t> & vec) const
  {
    return apply_re_sign(vec, rl_params_->observation_re_sign);
  };

  RLParameters * rl_params_;
  Observations obs_;

  DVec<tensor_element_t> obs_vec_;
  DVec<tensor_element_t> obs_history_vec_;
  DVec<tensor_element_t> raw_action_vec_;
  DVec<tensor_element_t> action_vec_;
  std::vector<long int> obs_term_dims_;
  std::vector<DVec<tensor_element_t>> obs_terms_;
  std::vector<DVec<tensor_element_t>> obs_term_history_vecs_;

  std::unique_ptr<InferrerBase> inferrer_;
  std::thread forward_thread;
  bool threadRunning;
  bool stop_update_ = false;
  bool thread_first_ = true;
  bool has_base_lin_vel_xy_observation_ = false;
  bool use_sim_base_lin_vel_source_ = false;
  std::mutex base_lin_vel_mutex_;
  std::mutex action_mutex_;
  std::mutex strict_log_mutex_;
  Vec3<tensor_element_t> latest_base_lin_vel_world_ = Vec3<tensor_element_t>::Zero();
  Vec3<tensor_element_t> latest_base_lin_vel_body_ = Vec3<tensor_element_t>::Zero();
  double last_base_lin_vel_sim_update_time_ = -1.0;
  double last_base_lin_vel_hw_update_time_ = -1.0;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr base_lin_vel_sim_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr base_lin_vel_hw_subscription_;

private:
  int iter_ = 0;
  double last_wheel_debug_time_ = 0.0;
  double last_leg_debug_time_ = 0.0;
  double last_obs_debug_time_ = 0.0;
  std::ofstream strict_start_log_;
  std::string strict_start_log_path_;
  size_t strict_policy_step_ = 0;
  static constexpr size_t kStrictPolicyLogLimit = 200;
};

#endif  // RL_CONTROLLER__FSM__FSMSTATE_RL_H_
