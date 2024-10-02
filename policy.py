from typing import Callable, Tuple

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class TraderMLPNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """

    def __init__(
        self,
        feature_dim: int,
        dim_pi: tuple[int] = (64, 64),
        dim_vf: tuple[int] = (64, 64),
        drop_out: float = 0.,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = dim_pi[-1]
        self.latent_dim_vf = dim_vf[-1]

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, dim_pi[0]), nn.Tanh(),
        )
        for prev_dim, dim in zip(dim_pi[:-1], dim_pi[1:]):
            self.policy_net.append(nn.Linear(prev_dim, dim))
            self.policy_net.append(nn.Tanh())
            if drop_out > 0:
                self.policy_net.append(nn.Dropout(drop_out))

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, dim_vf[0]), nn.Tanh(),
        )
        for prev_dim, dim in zip(dim_vf[:-1], dim_vf[1:]):
            self.value_net.append(nn.Linear(prev_dim, dim))
            self.value_net.append(nn.Tanh())
            if drop_out > 0:
                self.value_net.append(nn.Dropout(drop_out))

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class TraderConvNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor
    :param ts_len: length of the time series part in the observation vector
    :param conv_channels: (int) number of channels for each 1D conv layer
    :param conv_kernels: (int) kernels for each 1D conv layer. Must match with conv_channels
    :param mlp_dims: dimension of each MLP layer, applied after the conv layers
    """

    def __init__(
            self,
            feature_dim: int,
            ts_len: int,
            conv_channels: tuple[int] = (64, 32),
            conv_kernels: tuple[int] = (2, 2),
            mlp_dims: tuple[int] = (64,),
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = mlp_dims[-1]
        self.latent_dim_vf = mlp_dims[-1]

        self.feature_dim = feature_dim
        self.ts_len = ts_len
        self.non_ts_len = feature_dim - ts_len

        # Policy network
        self.conv = nn.Sequential()
        conv_dims = [1] + list(conv_channels)
        for (c_in, c_out), k in zip(zip(conv_dims[:-1], conv_dims[1:]), conv_kernels):
            self.conv.append(nn.Conv1d(c_in, c_out, k))
            self.conv.append(nn.Tanh())

        mlp_inp_dim = self.non_ts_len + conv_channels[-1]
        mlp_dims = [mlp_inp_dim] + list(mlp_dims)
        self.mlp_policy = nn.Sequential()
        self.mlp_value = nn.Sequential()
        for dim_in, dim_out in zip(mlp_dims[:-1], mlp_dims[1:]):
            self.mlp_policy.append(nn.Linear(dim_in, dim_out))
            self.mlp_policy.append(nn.Tanh())
            self.mlp_value.append(nn.Linear(dim_in, dim_out))
            self.mlp_value.append(nn.Tanh())

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        ts_embeds = self.conv(features[:, -self.ts_len:].unsqueeze(1)).mean(dim=-1)
        concat = th.cat((features[:, :self.non_ts_len], ts_embeds), dim=1)
        latent = self.mlp_policy(concat)
        return latent

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        ts_embeds = self.conv(features[:, -self.ts_len:].unsqueeze(1)).mean(dim=-1)
        concat = th.cat((features[:, :self.non_ts_len], ts_embeds), dim=1)
        latent = self.mlp_value(concat)
        return latent


class TraderActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: dict = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self._net_arch = net_arch or {}
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # self.mlp_extractor = TraderMLPNetwork(self.features_dim, **self._net_arch)
        self.mlp_extractor = TraderConvNetwork(self.features_dim, **self._net_arch)
