import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
from hemidiff.model.perception.sensory_encoder import BaseSensoryEncoder
from hemidiff.model.common.normalizer import LinearNormalizer
from hemidiff.policy.consensus_base_policy import ConsensusBasePolicy
from hemidiff.model.diffusion.conditional_unet1d import ConditionalUnet1D
from hemidiff.common.pytorch_util import dict_apply

from typing import List, Dict, Optional, Tuple

class ConsensusRouterPolicy(ConsensusBasePolicy):
    def __init__(self, 
        shape_meta: dict,
        num_modules: int,
        noise_scheduler: DDPMScheduler,
        obs_encoders: List[BaseSensoryEncoder],
        horizon, 
        n_action_steps, 
        n_obs_steps,
        num_inference_steps: Optional[int] = None,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int] = (128,256,512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        composition_strategy: str = "soft_gating",
        topk: int = 2,
        asymmetric_routing: bool = False,
        action_dim: Optional[int] = None, 
        **kwargs
    ):
        super().__init__()

        # Modalities sorting
        preferred_order = ['rgb', 'depth', 'pointcloud', 'dino_pointcloud']
        modalities_set = {
            modality
            for encoder in obs_encoders
            for modality in encoder.modalities()
        }
        ordered = [m for m in preferred_order if m in modalities_set]
        remaining = sorted(modalities_set - set(ordered))
        modalities = ['robot_state', *ordered, *remaining]

        # Action Dim Setup
        action_shape = shape_meta['action']['shape']
        if action_dim is not None:
            self.action_dim = action_dim
        else:
            self.action_dim = action_shape[0]

        # Obs config parsing
        obs_shape_meta = shape_meta['obs']
        obs_config = {key: [] for key in modalities}
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_config[type].append(key)
                obs_ports.append(key)

        # Create Encoder
        obs_encoder = self.create_observation_encoder(
            obs_encoders=obs_encoders,
            modalities=modalities,
            obs_config=obs_config,
            obs_key_shapes=obs_key_shapes,
        )

        obs_feature_dim = obs_encoder.output_feature_dim()
        total_global_dim = sum(obs_feature_dim.values()) * n_obs_steps
        models = nn.ModuleList()
        for _ in range(num_modules):
            models.append(
                ConditionalUnet1D(
                    input_dim=self.action_dim,
                    local_cond_dim=None,
                    global_cond_dim=total_global_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                )
            )

        self.asymmetric_routing = asymmetric_routing
        out_dim = len(models) * self.action_dim if asymmetric_routing else len(models)

        # Router
        weight_predictor = nn.Sequential(
            nn.Linear(total_global_dim, total_global_dim),
            nn.ReLU(),
            nn.Linear(total_global_dim, out_dim),
        )

        # Assignments
        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_config = obs_config
        self.obs_encoder = obs_encoder
        self.models = models
        self.weight_predictor = weight_predictor
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_modules = num_modules
        self.composition_strategy = composition_strategy
        self.topk = topk
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print(f"Initialized {self.get_policy_name()}. Global Dim: {total_global_dim}, Action Dim: {self.action_dim}, Modules: {num_modules}")

    def get_noise_prediction_network(self) -> nn.Module:
        return self

    def get_observation_encoder(self) -> BaseSensoryEncoder:
        return self.obs_encoder

    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    
    def get_observation_ports(self) -> List[str]:
        return self.obs_ports

    def get_policy_name(self) -> str:
        return 'consensus_router'

    def create_dummy_observation(self, batch_size=1, device=None):
        return super().create_dummy_observation(batch_size, self.n_obs_steps, self.obs_key_shapes, device)
    
    def predict_noise(self, trajectory, timesteps, global_cond):
        weights = self.weight_predictor(global_cond)
        
        batch_size = trajectory.shape[0]
        current_num_models = len(self.models)
        
        if self.composition_strategy == "soft_gating":
            if self.asymmetric_routing:
                # Weights: (B, M*D) -> (B, M, 1, D)
                w_reshaped = weights.reshape(batch_size, current_num_models, self.action_dim)
                w_reshaped = w_reshaped.unsqueeze(2) 
                norm_weights = F.softmax(w_reshaped, dim=1)
            else:
                norm_weights = F.softmax(weights, dim=1)

        if self.composition_strategy == "soft_gating":
            if self.asymmetric_routing:
                pred = sum(
                    norm_weights[:, i] * model(trajectory, timesteps, global_cond=global_cond)
                    for i, model in enumerate(self.models)
                )
            else:
                pred = sum(
                    norm_weights[:, i, None, None] * model(trajectory, timesteps, global_cond=global_cond)
                    for i, model in enumerate(self.models)
                )
        else:
            raise ValueError(f"Unknown strategy: {self.composition_strategy}")
            
        return pred

    def forward(self, sample, timestep, global_cond=None, **kwargs):
        return self.predict_noise(sample, timestep, global_cond)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_list = self.normalization(obs_dict)
        value = next(iter(nobs_list[0].values()))
        B = value.shape[0]
        
        features = self.obs_encoder(dict_apply(nobs_list[0], lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])))
        features = dict_apply(features, lambda x: x.reshape(B, -1))
        global_cond = torch.cat(list(features.values()), dim=-1)

        trajectory = torch.randn(
            size=(B, self.horizon, self.action_dim),
            device=global_cond.device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.predict_noise(trajectory, t, global_cond)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory).prev_sample

        action_pred = self.normalizer['action'].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {'action': action, 'action_pred': action_pred, 'weights': self.weight_predictor(global_cond)}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        features = self.obs_encoder(
            dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        )
        features = dict_apply(features, lambda x: x.reshape(batch_size, -1))
        
        global_cond = torch.cat(list(features.values()), dim=-1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        pred = self.predict_noise(noisy_trajectory, timesteps, global_cond)

        loss = F.mse_loss(pred, noise)
        return loss
    
    def adapt(self, method='weight_predictor+obs_encoder'):
        pass
    
    def conditional_sample(self, *args, **kwargs): pass
