import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from nav_baseline.common.aux_losses import AuxLosses
from nav_baseline.models.resnet_encoders import (
    RGBEncoderResnet50,
    DepthEncoderResnet50,
)
from nav_baseline.models.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from nav_baseline.models.policy import BasePolicy

class Seq2SeqPolicy(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class Seq2SeqNet(Net):
    r"""
    A baseline sequence to sequence network that concatenates, RGB, and depth 
    encodings before decoding an action distribution with an RNN.

    Modules:
        Depth encoder
        RGB encoder
        RNN state encoder
    """
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions
    ):
        super().__init__()
        self.model_config = model_config

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in model_config.DEPTH_ENCODER.supported_encoders, \
            f"DEPTH_ENCODER.cnn_type must be in {model_config.DEPTH_ENCODER.supported_encoders}"
            
        if model_config.DEPTH_ENCODER.cnn_type == "DepthEncoderResnet50":
            self.depth_encoder = DepthEncoderResnet50(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in model_config.RGB_ENCODER.supported_encoders, \
            f"RGB_ENCODER.cnn_type must be in {model_config.RGB_ENCODER.supported_encoders}"

        if model_config.RGB_ENCODER.cnn_type == "RGBEncoderResnet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = RGBEncoderResnet50(
                observation_space, 
                model_config.RGB_ENCODER.output_size, 
                device
            )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        
        # Init the RNN state decoder
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )
        
        if model_config.SEQ2SEQ.use_pointgoal:
            rnn_input_size += (
                observation_space.spaces["pointgoal_with_gps_compass"].shape[0]
            )
        if model_config.SEQ2SEQ.use_heading:
            rnn_input_size += (
                observation_space.spaces["heading"].shape[0]
            )
            
        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        
        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        x = torch.cat([depth_embedding, rgb_embedding], dim=1)
        
        if self.model_config.SEQ2SEQ.use_pointgoal:
            pointgoal_embedding = observations["pointgoal_with_gps_compass"]
            x = torch.cat([x, pointgoal_embedding], dim=1)
        
        if self.model_config.SEQ2SEQ.use_heading:
            heading_embedding = observations["heading"]
            x = torch.cat([x, heading_embedding])

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        # if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
        #     progress_hat = torch.tanh(self.progress_monitor(x))
        #     progress_loss = F.mse_loss(
        #         progress_hat.squeeze(1), observations["progress"], reduction="none"
        #     )
        #     AuxLosses.register_loss(
        #         "progress_monitor",
        #         progress_loss,
        #         self.model_config.PROGRESS_MONITOR.alpha,
        #     )

        return x, rnn_hidden_states
