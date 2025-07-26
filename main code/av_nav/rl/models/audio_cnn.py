# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from av_nav.common.utils import Flatten

#AudioCNN类是一个用于处理音频数据的CNN模型，它可以嵌入音频特征，以便在强化学习或其他机器学习任务中使用。
class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor):
        #初始化函数
        super().__init__()
        #调用父类的构造函数
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        #读取音频通道数
        self._audiogoal_sensor = audiogoal_sensor
        #保存音频传感器的名称


        cnn_dims = np.array(
            observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.float32
        )
        #获取cnn的输入维度（高度和宽度）
        #observation_space参数是一个配置对象，它描述了代理的观察空间。output_size是CNN输出嵌入向量的大小，audiogoal_sensor是观察空间中音频数据的键名


        #计算每层的输出尺寸
        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        #定义cnn层定义了一个由三个卷积层、两个ReLU激活函数、一个展平层和一个全连接层组成的CNN模型
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            # nn.ReLU(True),
            # nn.Conv2d(
            #     in_channels=64,
            #     out_channels=32,
            #     kernel_size=self._cnn_layers_kernel_size[3],
            #     stride=self._cnn_layers_stride[3],
            # ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )#这部分代码根据输入尺寸动态调整CNN层的核大小和步长，并计算每层之后的输出尺寸

        #初始化网络层的权重
        self.layer_init()

    #方法根据输入尺寸、填充、扩张、核大小和步长计算卷积层的输出尺寸
    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    #layer_init方法使用Kaiming初始化方法来初始化卷积层和全连接层的权重，并将偏置初始化为0
    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    #forward方法定义了数据通过网络的前向传播过程。它从输入观察中提取音频数据，调整其维度，然后将其传递给CNN模型以产生嵌入向量
    def forward(self, observations):
        cnn_input = []

        audio_observations = observations[self._audiogoal_sensor]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
      

        return cnn_input
