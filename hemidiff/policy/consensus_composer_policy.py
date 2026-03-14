import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import copy

from hemidiff.policy.policy_base import BasePolicy
from hemidiff.model.perception.sensory_encoder import BaseSensoryEncoder
from hemidiff.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from hemidiff.common.pytorch_util import dict_apply

from typing import List, Dict, Optional, Union

class ConsensusComposerPolicy(BasePolicy):
    def __init__(self,
        shape_meta: Dict,
        noise_scheduler: DDPMScheduler,
        modular_policies: List[BasePolicy],
        policy_weights: List[float],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        active_slices: Optional[List[List[int]]] = None,
        **kwargs
    ):
        super().__init__()

        self.active_slices = [slice(s[0], s[1]) for s in active_slices] if active_slices else None
        if self.active_slices:
            assert len(self.active_slices) == len(modular_policies)

        modalities = list(set(['robot_state', *[
            modality 
            for policy in modular_policies
            for modality in policy.get_observation_modalities()
        ]]))

        action_shape = shape_meta["action"]["shape"]
        action_dim = action_shape[0]
        obs_ports = list(set([
            port 
            for policy in modular_policies
            for port in policy.get_observation_ports()
        ]))

        obs_encoder = self.create_observation_encoder(
            obs_encoders=[policy.get_observation_encoder() for policy in modular_policies],
            modalities=modalities
        )

        self.normalizer = LinearNormalizer()

        def normalization(obs_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
            nobs_list = []
            for policy in modular_policies:
                this_obs = {port: obs_dict[port] for port in policy.get_observation_ports()}
                this_nobs = policy.normalizer.normalize(this_obs)
                nobs_list.append(this_nobs)
            return nobs_list

        def unnormalization(action: torch.Tensor):
            return self.normalizer['action'].unnormalize(action)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps

        self.normalization = normalization
        self.unnormalization = unnormalization
        self.modalities = modalities
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.policy_weights = policy_weights
        self.modular_policies = nn.ModuleList(modular_policies)
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_feature_dims = obs_encoder.output_feature_dim()
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        print(f"Initialized Compositional Policy. Active Slices: {self.active_slices}")

    def get_noise_prediction_network(self) -> nn.ModuleList:
        return [policy.get_noise_prediction_network() for policy in self.modular_policies]

    def get_observation_encoder(self) -> BaseSensoryEncoder:
        return self.obs_encoder
    
    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    
    def get_observation_ports(self) -> List[str]:
        return self.obs_ports

    def get_policy_name(self) -> str:
        return 'consensus_composer'

    def conditional_sample(self,
        global_conds: List[torch.Tensor], 
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> torch.Tensor:
        pass

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trajectory = torch.randn(
            size=(obs_dict[list(obs_dict.keys())[0]].shape[0], self.horizon, self.action_dim),
            device=obs_dict[list(obs_dict.keys())[0]].device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        sub_policy_conds = []
        all_policy_weights = [] 
        for policy in self.modular_policies:
            this_obs = {port: obs_dict[port] for port in policy.get_observation_ports()}
            this_nobs = policy.normalizer.normalize(this_obs)
            B = this_nobs[list(this_nobs.keys())[0]].shape[0]
            features = policy.obs_encoder(
                dict_apply(this_nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            )
            features = dict_apply(features, lambda x: x.reshape(B, -1))
            global_cond = torch.cat(list(features.values()), dim=-1)
            sub_policy_conds.append(global_cond)

            with torch.no_grad():
                logits = policy.weight_predictor(global_cond) 
                b_size = logits.shape[0]
                num_mods = policy.num_modules
                act_dim = policy.action_dim
                
                if policy.asymmetric_routing:
                    w = logits.reshape(b_size, num_mods, act_dim)
                    probs = torch.nn.functional.softmax(w, dim=1) 
                    avg_probs = probs.mean(dim=2) 
                else:
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    avg_probs = probs

                all_policy_weights.append(avg_probs)

        for t in self.noise_scheduler.timesteps:
            total_noise = torch.zeros_like(trajectory)
            
            for i, (policy, cond) in enumerate(zip(self.modular_policies, sub_policy_conds)):
                if self.active_slices:
                    model_input = trajectory[..., self.active_slices[i]]
                else:
                    model_input = trajectory
                noise_pred = policy.predict_noise(model_input, t, cond)
                
                weight = self.policy_weights[i]
                if self.active_slices:
                    total_noise[..., self.active_slices[i]] += weight * noise_pred
                else:
                    total_noise += weight * noise_pred

            trajectory = self.noise_scheduler.step(total_noise, t, trajectory).prev_sample

        action_pred = self.unnormalization(trajectory)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start : end]

        return {
            'action': action,
            'action_pred': action_pred,
            'component_weights': all_policy_weights 
        }

    def compute_loss(self, batch):
        total_loss = 0.0
        
        for i, policy in enumerate(self.modular_policies):
            sub_batch = batch.copy()
            if self.active_slices is not None:
                sub_batch['action'] = batch['action'][..., self.active_slices[i]]
            
            loss = policy.compute_loss(sub_batch)
            total_loss += loss
            
        return total_loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if self.active_slices:
            for policy, slc in zip(self.modular_policies, self.active_slices):
                sub_norm = copy.deepcopy(normalizer)
                if 'action' in sub_norm.params_dict:
                    src = normalizer['action'].params_dict
                    new_action_norm = SingleFieldLinearNormalizer.create_manual(
                        scale=src['scale'][slc],
                        offset=src['offset'][slc],
                        input_stats_dict={k: v[slc] for k, v in src['input_stats'].items()}
                    )
                    sub_norm.params_dict['action'] = new_action_norm.params_dict
                policy.set_normalizer(sub_norm)
        else:
            for policy in self.modular_policies:
                policy.set_normalizer(normalizer)
