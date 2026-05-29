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

#ifndef RL_CONTROLLER__FSM__FSMSTATE_RLDREAMWAQ_H_
#define RL_CONTROLLER__FSM__FSMSTATE_RLDREAMWAQ_H_

#include <memory>
#include <string>
#include <vector>

#include "FSMState_RL.h"
#include "rl_controller/inferrer/inferrer_base.hpp"

class FSMState_RLDreamWaQ : public FSMState_RL
{
public:
  FSMState_RLDreamWaQ(
    std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName);
  virtual ~FSMState_RLDreamWaQ() {}

protected:
  void update_forward() override;

private:
  void validate_dreamwaq_config() const;
  void setup_dreamwaq_models();
  void validate_dreamwaq_model_inputs() const;
  std::vector<tensor_element_t> build_dreamwaq_actor_input(
    const std::vector<tensor_element_t> & estimated_velocity,
    const std::vector<tensor_element_t> & latent_mean) const;

  std::unique_ptr<InferrerBase> dreamwaq_encoder_inferrer_;
  std::unique_ptr<InferrerBase> dreamwaq_vel_mu_inferrer_;
  std::unique_ptr<InferrerBase> dreamwaq_latent_mu_inferrer_;
};

#endif  // RL_CONTROLLER__FSM__FSMSTATE_RLDREAMWAQ_H_
