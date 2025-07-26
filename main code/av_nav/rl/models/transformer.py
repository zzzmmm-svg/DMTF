import copy
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import av_nav.rl.models.configs as configs
from torch.nn.functional import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn import Flatten
from av_nav.rl.models.modeling_resnet import ResNetV2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    	
class Embedding(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embedding, self).__init__()
        
        self.hybrid = None
        in_channels=2
        img_size = 64
        img_size = (64,64)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #self.position_embeddings = nn.Parameter(torch.zeros(2, n_patches, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])



class Transformer(nn.Module):#定义一个transfermer模型

    def __init__(self, d_model=256, nhead=2, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=384, dropout=0.3,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        #d_model模型的嵌入维度，模型的输入和输出都将被映射到这个维度
        #nhead是多头注意力的头数，注意力机制多头
        #num_encoder_layers是编码器的层数，每层都包含一个子注意力模块和前馈网络
        #解码器类似，每层都包含一个子注意力模块和前馈神经网络，还有一个激活函数，线性变换
        #dim——forward这是前馈神经网络的输出维度Transformer中的每个前馈网络都是一个简单的线性变换，后跟一个激活函数，然后再跟一个线性变换
        #Dropout是一种正则化技术，用于防止过拟合，通过在训练过程中随机丢弃一些神经元的输出来实现。
        #默认情况下，使用的是ReLU激活函数，但也可以是其他激活函数，如GELU（Gaussian Error Linear Unit）或GLU（Gated Linear Unit）。
        #归一化（Layer Normalization）是在每个子层（自注意力或前馈网络）之前还是之后应用。如果设置为True，则归一化在每个子层之前应用；如果为False，则在每个子层之后应用。
        #return_intermediate_dec 参数控制解码器是否返回所有中间层的输出。如果设置为True，则解码器将返回一个包含所有层输出的列表；如果为False，则只返回最后一层的输出。

        #初始化函数
        super().__init__()
        #调用父类的构造函数
        #创建编码器实例
        
        
        

        self.embeddings = Embeddings(self.config)#bs*1*128*128


        
        #保留模型的维度和头数
        self.d_model = 256
        self.nhead = 2

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        self.encoder1 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        #创建一个transfermer柏宁玛琪
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)


        

        

    def transform_tensor(self,input_tensor):
    	assert input_tensor.size(1) == 512, "Input tensor's second dimension must be 512"

    	# 定义一个线性层，将输入从 512 维度映射到 256 维度
    	# 注意：这里我们不需要指定批量大小，因为线性层会自动处理任何批量大小
    	linear_layer = nn.Linear(in_features=512, out_features=256)

    	# 应用线性层
    	output_tensor = linear_layer(input_tensor)

    	return output_tensor
    def _reset_parameters(self):
        #初始化模型参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)






