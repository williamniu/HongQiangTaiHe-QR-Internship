import torch
import torch.nn as nn
from config import params

'''
    Func: 模型定义文件
'''

class MixerBlock(nn.Module):
    def __init__(self, time_steps, feature_dim, hidden_size):
        super(MixerBlock, self).__init__()
        self.time_norm = nn.LayerNorm([time_steps,feature_dim])
        self.time_mlp = nn.Sequential(
            nn.Linear(time_steps,hidden_size),
            nn.GELU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(hidden_size,time_steps)
        )

        self.feature_norm = nn.LayerNorm([time_steps,feature_dim])
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim,hidden_size),
            nn.GELU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(hidden_size,feature_dim)
        )

    def forward(self,x):
        # print(rf"x的张量维度为:{x.shape}")
        y = self.time_norm(x)
        y = y.transpose(1,2)
        y = self.time_mlp(y)
        y = y.transpose(1,2)
        x = x+y

        y = self.feature_norm(x)
        y = self.feature_mlp(y)

        x = x+y
        return x

class FactorTSMixer(nn.Module):
    def __init__(self, args):
        super(FactorTSMixer, self).__init__()
        self.lookback = params.lookback_window
        self.feature_dim = params.factor_num
        self.hidden_size = params.hidden_size
        self.num_layers = params.mixer_layers

        # Mixer blocks
        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(self.lookback, self.feature_dim, self.hidden_size)
              for _ in range(self.num_layers)]
        )

        # 最终回归层：取最后一个时间步所有因子，降维到一个输出值
        # self.head = nn.Linear(self.lookback * self.feature_dim, self.feature_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim,eps=1e-5),
            nn.Linear(self.feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tsdata):
        """
        tsdata: [batch, time, feature]
        """
        x = tsdata.float()
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"NaN or inf detected in model input: {x.shape}")
        x = self.mixer_blocks(x)
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"NaN or inf after mixer_blocks: {x.shape}")
        x = x[:, -1, :]
        out = self.head(x)
        # out = 10 * torch.tanh(out)
        if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
            print(f"NaN or inf in model output: {out.shape}")
        return out
        

