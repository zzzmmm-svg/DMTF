#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torch.nn as nn
from torchsummary import summary

from av_nav.common.utils import CategoricalNet
from av_nav.rl.models.rnn_state_encoder import RNNStateEncoder
from av_nav.rl.models.visual_cnn import VisualCNN
from av_nav.rl.models.audio_cnn import AudioCNN
from av_nav.rl.models.transformer import Transformer

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """
    #
    def __init__(self, observation_space,hidden_size, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        
        
        self.goal_sensor_uuid = goal_sensor_uuid
        #传入目标传感器
        self._hidden_size = hidden_size
        #传入隐藏层大小
        self._audiogoal = False
        #初始化音频目标，点目标的标志、点慕目标数量
        self._pointgoal = False
        self._n_pointgoal = 0
        #self.pos =  torch.zeros(64,5,256)
        
        hidden_dim = 256
        deep_supervision= False
        # Transformer parameters:
        nheads = 2
        dropout = 0.5
        dim_feedforward = 384
        enc_layers = 6
        dec_layers = 6
        pre_norm = False
        
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer.to(device)
        
        
        

        #目标传感器UUID判断是否使用音频目标、点目标，并设置相应的标志和点目标数量。
        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True #同时音频和点目标
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        #视觉编码器实例，用于处理视觉输入
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        #额外的参数用于处理rgb图像
        #使用音频目标，创建音频编码器实例，用于处理音频输入
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)



        #RNN的输入大小，并创建状态编码器实例，用于处理RNN输入
        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)
        #

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        #定义模型进行前向传播

        #空列表，用于存储不同编码器的输出
        x = []



        #对于replica的audio_depth专项
        if self._audiogoal and not self.is_blind:
        	
        	x.append(self.transformer(self.visual_encoder(observations), self.audio_encoder(observations)))
        	
        	#x1 = x1.flatten()
        	
        	#print(x1)
        	#x= torch.unbind(x, dim=1)  # 将x沿着dim=1解包成列表
        	#x.append(x1)
        else:
            #如果有点目标
            if self._pointgoal:
                x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
            if self._audiogoal:
                x.append(self.audio_encoder(observations))
            if not self.is_blind:
                x.append(self.visual_encoder(observations))

        #print(len(x))1




        x1 = torch.cat(x,dim=1)
        #print(x1.shape)torch.Size([5, 1024])
        #print(x1.shape)
        #沿着第一个维度dim=1，链接起来
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        # print(x2.shape)
        # print(rnn_hidden_states1.shape)
        #print(x2.shape)torch.Size([5, 512])
        #print(rnn_hidden_states1.shape)torch.Size([1, 5, 512])
        assert not torch.isnan(x2).any().item()
        #RNN的输出x2和更新后的隐藏状态rnn_hidden_states1
        return x2, rnn_hidden_states1
