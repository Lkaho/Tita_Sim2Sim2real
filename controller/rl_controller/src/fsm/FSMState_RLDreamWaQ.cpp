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

#include "rl_controller/fsm/FSMState_RLDreamWaQ.h"

#include <iostream>
#include <stdexcept>

#include "rl_controller/common/timeMarker.h"

#ifdef USE_ENGINE
#include "rl_controller/inferrer/engine_inferrer.hpp"
#else
#include "rl_controller/inferrer/onnx_inferrer.hpp"
#endif

namespace
{
std::unique_ptr<InferrerBase> make_dreamwaq_inferrer()
{
#ifdef USE_ENGINE
  throw std::runtime_error(
    "[FSMState_RLDreamWaQ] DreamWaQ/CENet split deployment supports ONNX runtime only; "
    "rebuild rl_controller with USE_ENGINE=OFF");
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
        "[FSMState_RLDreamWaQ] " + model_label + " input " + std::to_string(input_index) +
        " has unsupported dynamic/invalid dimension: " + std::to_string(dim));
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

void validate_single_input_dim(
  const InferrerBase & inferrer, const std::string & model_label, size_t expected_dim)
{
  const auto & input_shapes = inferrer.getInputShapes();
  if (input_shapes.size() != 1) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] " + model_label + " model must have one input, got " +
      std::to_string(input_shapes.size()));
  }

  const size_t model_dim = tensor_element_count(input_shapes[0], model_label, 0);
  if (model_dim != expected_dim) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] " + model_label + " input dim mismatch, configured=" +
      std::to_string(expected_dim) + ", model=" + std::to_string(model_dim));
  }
}
}  // namespace

FSMState_RLDreamWaQ::FSMState_RLDreamWaQ(
  std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName)
: FSMState_RL(data, rl_params, stateName)
{
  validate_dreamwaq_config();
  setup_dreamwaq_models();
  validate_dreamwaq_model_inputs();
}

void FSMState_RLDreamWaQ::validate_dreamwaq_config() const
{
#ifdef USE_ENGINE
  throw std::runtime_error(
    "[FSMState_RLDreamWaQ] DreamWaQ/CENet split deployment supports ONNX runtime only; "
    "rebuild rl_controller with USE_ENGINE=OFF");
#endif

  if (rl_params_->use_velocity_estimator) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] use_velocity_estimator must be false; DreamWaQ uses its own "
      "CENet velocity head");
  }
  if (!use_term_history_layout()) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] observations_history_mode must be \"term\" for exported CENet "
      "history layout");
  }
  if (rl_params_->history_len <= 0) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] history_len must be positive");
  }
  if (rl_params_->dreamwaq_encoder_policy_path.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_encoder_policy_path is required");
  }
  if (rl_params_->dreamwaq_vel_mu_policy_path.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_vel_mu_policy_path is required");
  }
  if (rl_params_->dreamwaq_latent_mu_policy_path.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_latent_mu_policy_path is required");
  }
  if (rl_params_->dreamwaq_encoder_output_name.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_encoder_output_name is required");
  }
  if (rl_params_->dreamwaq_vel_mu_output_name.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_vel_mu_output_name is required");
  }
  if (rl_params_->dreamwaq_latent_mu_output_name.empty()) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] dreamwaq_latent_mu_output_name is required");
  }
  if (rl_params_->dreamwaq_encoder_feature_dim <= 0 || rl_params_->dreamwaq_velocity_dim <= 0 ||
      rl_params_->dreamwaq_latent_dim <= 0) {
    throw std::runtime_error("[FSMState_RLDreamWaQ] DreamWaQ dimensions must be positive");
  }
}

void FSMState_RLDreamWaQ::setup_dreamwaq_models()
{
  dreamwaq_encoder_inferrer_ = make_dreamwaq_inferrer();
  dreamwaq_vel_mu_inferrer_ = make_dreamwaq_inferrer();
  dreamwaq_latent_mu_inferrer_ = make_dreamwaq_inferrer();

  dreamwaq_encoder_inferrer_->loadModel(rl_params_->dreamwaq_encoder_policy_path);
  dreamwaq_encoder_inferrer_->setOutput(
    rl_params_->dreamwaq_encoder_output_name, rl_params_->dreamwaq_encoder_feature_dim);

  dreamwaq_vel_mu_inferrer_->loadModel(rl_params_->dreamwaq_vel_mu_policy_path);
  dreamwaq_vel_mu_inferrer_->setOutput(
    rl_params_->dreamwaq_vel_mu_output_name, rl_params_->dreamwaq_velocity_dim);

  dreamwaq_latent_mu_inferrer_->loadModel(rl_params_->dreamwaq_latent_mu_policy_path);
  dreamwaq_latent_mu_inferrer_->setOutput(
    rl_params_->dreamwaq_latent_mu_output_name, rl_params_->dreamwaq_latent_dim);

  std::cout << "[FSMState_RLDreamWaQ] split CENet enabled: actor=" << rl_params_->policy_path
            << ", encoder=" << rl_params_->dreamwaq_encoder_policy_path
            << ", vel_mu=" << rl_params_->dreamwaq_vel_mu_policy_path
            << ", latent_mu=" << rl_params_->dreamwaq_latent_mu_policy_path << std::endl;
}

void FSMState_RLDreamWaQ::validate_dreamwaq_model_inputs() const
{
  validate_single_input_dim(*dreamwaq_encoder_inferrer_, "cenet_encoder", actor_history_input_dim());
  validate_single_input_dim(
    *dreamwaq_vel_mu_inferrer_, "cenet_vel_mu",
    static_cast<size_t>(rl_params_->dreamwaq_encoder_feature_dim));
  validate_single_input_dim(
    *dreamwaq_latent_mu_inferrer_, "cenet_latent_mu",
    static_cast<size_t>(rl_params_->dreamwaq_encoder_feature_dim));
  validate_single_input_dim(
    *inferrer_, "cenet_actor",
    static_cast<size_t>(
      rl_params_->num_obs + rl_params_->dreamwaq_velocity_dim + rl_params_->dreamwaq_latent_dim));
}

std::vector<tensor_element_t> FSMState_RLDreamWaQ::build_dreamwaq_actor_input(
  const std::vector<tensor_element_t> & estimated_velocity,
  const std::vector<tensor_element_t> & latent_mean) const
{
  if (estimated_velocity.size() != static_cast<size_t>(rl_params_->dreamwaq_velocity_dim)) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] estimated velocity output size mismatch, expected " +
      std::to_string(rl_params_->dreamwaq_velocity_dim) + ", got " +
      std::to_string(estimated_velocity.size()));
  }
  if (latent_mean.size() != static_cast<size_t>(rl_params_->dreamwaq_latent_dim)) {
    throw std::runtime_error(
      "[FSMState_RLDreamWaQ] latent mean output size mismatch, expected " +
      std::to_string(rl_params_->dreamwaq_latent_dim) + ", got " +
      std::to_string(latent_mean.size()));
  }

  std::vector<tensor_element_t> actor_input = eigenToVector(obs_vec_);
  actor_input.reserve(
    actor_input.size() + static_cast<size_t>(rl_params_->dreamwaq_velocity_dim) +
    static_cast<size_t>(rl_params_->dreamwaq_latent_dim));
  actor_input.insert(actor_input.end(), estimated_velocity.begin(), estimated_velocity.end());
  actor_input.insert(actor_input.end(), latent_mean.begin(), latent_mean.end());
  return actor_input;
}

void FSMState_RLDreamWaQ::update_forward()
{
  const long long interval = static_cast<long long>(rl_params_->time_interval * 1000000);
  while (threadRunning) {
    long long _start_time = getSystemTime();

    if (!stop_update_) {
      update_observations();
      append_observation_history();

      std::vector<std::vector<tensor_element_t>> encoder_inputs;
      encoder_inputs.push_back(eigenToVector(obs_history_vec_));
      const auto encoder_feature = dreamwaq_encoder_inferrer_->computeActions(encoder_inputs);
      const auto estimated_velocity = dreamwaq_vel_mu_inferrer_->computeActions({encoder_feature});
      const auto latent_mean = dreamwaq_latent_mu_inferrer_->computeActions({encoder_feature});

      estimated_base_lin_vel_body_[0] = estimated_velocity[0];
      estimated_base_lin_vel_body_[1] = estimated_velocity[1];
      estimated_base_lin_vel_body_[2] =
        estimated_velocity.size() > 2 ? estimated_velocity[2] : static_cast<tensor_element_t>(0.0);

      auto actor_input = build_dreamwaq_actor_input(estimated_velocity, latent_mean);
      auto raw_actions = vectorToEigen(inferrer_->computeActions({actor_input}));
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
    }
    absoluteWait(_start_time, interval);
  }
  threadRunning = false;
}
