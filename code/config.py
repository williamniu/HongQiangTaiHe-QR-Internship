import os
import torch
import pandas as pd
from argparse import ArgumentParser

class Config:
    # 路径配置
    root_path = rf'/home/user45/NZH'       # 模型文件存储目录
    data_path = rf'/project/model_share/share_1'    # 数据存储目录（share_1, share_2等）
    fac_path, fac_name = rf'{data_path}/factor_data', rf'fac20240819'   # 因子
    label_path, label_name = rf'{data_path}/label_data', rf'label1'     # 标签
    liquid_path, liquid_name = rf'{data_path}/label_data', rf'can_trade_amt1'   # 流动性数据

    # 其他配置
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.autograd.set_detect_anomaly(True)
    
# 超参数设置
def parse_args():
    parser = ArgumentParser()
    # 添加传入的位置参数
    parser.add_argument('gpus', type=str, default=0, help='GPU IDs to use')
    parser.add_argument('valid_period_idx', type=str, default=1, help='Validation period index')
    
    parser.add_argument('--max_epochs', type=int, default=30)  # 最大轮数
    parser.add_argument('--batch_size', type=int, default=2)  # batch中样本含几天
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.002)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=1e-2)  # weight_decay
    parser.add_argument('--seed', type=int, default=3253)  # 随机种子
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adamw', 'adamw'])
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--loss', default='wpcc')  # 损失函数
    parser.add_argument('--early_stop', action='store_true')  # 早停设置
    parser.add_argument('--swa', action='store_true')  # swa设置
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')
    
    return parser.parse_args()


class params:
    model_path = rf'{Config.root_path}/model_train'
    profiler_path = rf'{Config.root_path}/logs'
    
    # 模型前缀
    model_prefix = rf'TSMixer'
    liquid_data = pd.read_feather(rf"{Config.liquid_path}/{Config.liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{Config.label_path}/{Config.label_name}.fea").set_index("index")
    
    # 是否使用dropout
    dropout = True
    # dropout率
    dropout_rate = 0.1

    # TSMixer特定参数设置
    lookback_window = 5
    hidden_size = 64
    mixer_layers = 3

    # 因子标准化方法
    normed_method = 'zscore'
    
    
def get_basic_name():
    name = rf'{params.model_prefix}--{Config.fac_name}--{Config.label_name}'
    if params.dropout:
        name += rf'--dropout{params.dropout_rate}'
    return name


if __name__ == "__main__":
    config = Config()
    args = parse_args()