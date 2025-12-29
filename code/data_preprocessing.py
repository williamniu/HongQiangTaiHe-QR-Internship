import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config import params


# 因子标准化
def normed_data(data, date_list, stage, normed_method=params.normed_method):
    """
    对连续日期窗口进行因子标准化，输出训练输入张量和目标标签。

    Args:
        data: pd.DataFrame, 包含多个日期的所有股票因子数据
        date_list: List[str], 时间窗口 + 标签日期 len = lookback + 1
        stage: 'train' | 'valid' | 'test'
        normed_method: str, 默认为 'zscore'

    Returns:
        X_seq: List[Tensor[N, F]]，共 T 个 T = lookback
        y:     Tensor[N]，滑窗后一天的收益率标签
        code:  np.ndarray[str]，股票代码
        liquid: Tensor[N]，滑窗最后一天的流动性
    """
    # 获取每一天的股票代码交集，确保时间序列对齐
    stock_sets = [set(data.loc[d]['Code']) for d in date_list]
    common_codes = list(set.intersection(*stock_sets))
    common_codes = sorted(set.intersection(*stock_sets))

    if len(common_codes) < 500:
        print(f"Skipping date {date_list[-1]} due to insufficient common_codes: {len(common_codes)}")
        return [], [], [], []
    
    X_seq = []
    for date in date_list[:-1]:  # 滑动窗口内的 T 天
        data_day = data.loc[date].reset_index()
        data_day = data_day[data_day['Code'].isin(common_codes)]

        data_day = data_day.reindex(
            data_day[params.factor_list].dropna(thresh=int(0.1 * len(params.factor_list))).index
        )
        data_day = data_day.set_index("date")

        # 合并流动性
        liquid_data = params.liquid_data.loc[date]
        data_day["liquid"] = liquid_data.reindex(data_day["Code"]).values

        # # 合并标签
        ret_data = params.ret_data.loc[date]
        data_day["Label"] = ret_data.reindex(data_day["Code"]).values
        # if stage == "train":
        #     data_day["Label"] = (data_day["Label"] - data_day["Label"].mean()) / data_day["Label"].std()
        # data_day["Label"] = data_day["Label"].fillna(0)

        code_value = data_day["Code"].values
        data_X = data_day.drop(['Code', 'Label', 'liquid'], axis=1)

        if normed_method == 'zscore':
            data_X = data_X.fillna(0)
            data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995), axis=1)
            mean = data_X.mean()
            std = data_X.std()
            std = std.where(std>1e-8,1.0)
            data_X = (data_X - mean) / std
            data_X = data_X.fillna(0)
            if np.any(np.isnan(data_X)) or np.any(np.isinf(data_X)):
                print(f"Warning: NAN or inf detected in data_X for date {date}")    
        else:
            raise NotImplementedError

        X_seq.append(torch.tensor(data_X.values, dtype=torch.float32))

    # 滑窗后一天（第 T+1
    label_date = date_list[-1]
    ret_data = params.ret_data.loc[label_date]
    label_data = ret_data.reindex(sorted(common_codes)).values

    # if stage=="train":
    #     mean = np.nanmean(label_data)
    #     std  = np.nanstd(label_data)
    #     if std < 1e-8:
    #         label_data = label_data - mean
    #     else:
    #         label_data = (label_data - mean) / std
    label_data = np.nan_to_num(label_data, nan=0, posinf=0, neginf=0)
    data_y = torch.tensor(label_data, dtype=torch.float32)

    # 使用最后一个输入日的流动性
    liquid_data = params.liquid_data.loc[date_list[-1]]
    liquid_values = liquid_data.reindex(sorted(common_codes)).values
    if np.any(np.isinf(liquid_values)):
        print(f"Warning: inf detected in liquid_values for date {date_list[-2]}")
    liquid_data = np.nan_to_num(liquid_values,nan = 0, posinf = 0, neginf = 0)
    data_liquid = torch.tensor(liquid_data, dtype=torch.float32)


    return X_seq, data_y, code_value, data_liquid


#返回的x_seq: 是一个列表： 长度为10（即lookback_window），每个张量维度是【stock_num,factor_num】


# batch 组合方法
def collate_fn(batch):
    """
    将 batch_size 个“市场级样本”拼接为一个大批次的 [B*N, T, F]
    """
    # print(rf"每个batch的维度是：{len(batch)}")
    # for item in batch:
    #     print(rf"每个batch中item的维度是：{item[0].shape}")
    # 每个 item 是一个 (data_X, data_y, data_time, code_value, data_liquid)
    data_X_list = [item[0] for item in batch]        # [N, T, F]
    data_y_list = [item[1] for item in batch]        # [N]
    data_time = [item[2] for item in batch]     # str 或 list[str]
    code_list = [np.array(item[3]).flatten() for item in batch]
    code_value = np.concatenate(code_list,axis = 0)        # np.ndarray[str]
    data_liquid_list = [item[4] for item in batch]   # [N]

    # 拼接：多个 [N, T, F] → 合成 [B*N, T, F]
    data_X = torch.cat(data_X_list, dim=0)
    data_y = torch.cat(data_y_list, dim=0)
    data_liquid = torch.cat(data_liquid_list, dim=0)

    for i, (x, y, liquid) in enumerate(zip(data_X_list, data_y_list, data_liquid_list)):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"NaN or inf detected in data_X_list[{i}]")
        if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            print(f"NaN or inf detected in data_y_list[{i}]")
        if torch.any(torch.isnan(liquid)) or torch.any(torch.isinf(liquid)):
            print(f"NaN or inf detected in data_liquid_list[{i}]")
    # print(rf"每个batch中的x的维度是:{data_X.shape}")
    # data_time = []
    # for i in range(len(data_time_list)):
    #     n_i = data_y_list[i].shape[0]
    #     t_i = data_time_list[i]
    #     data_time.extend([t_i]*n_i)
    return data_X, data_y, data_time, code_value, data_liquid




# 统一数据框架
# Dataset的作用是整理输出单个样本数据
# dataloader的作用是承接dataset中单个样本的数据并返回一个batch（批次）的训练数据
class DLDataset(Dataset):
    def __init__(self, date_list, stage='train'):
        self.date_list = date_list
        self.stage = stage
        self.lookback = params.lookback_window

    def __getitem__(self, index):
        if index + self.lookback >= len(self.date_list):
            return ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample']

        date_seq = self.date_list[index : index + self.lookback + 1]
        # input_dates = date_seq[:-1]
        label_date = date_seq[-1]

        data = params.all_data.loc[date_seq].copy()
        train_X, train_y, code_value, train_liquid = normed_data(data, date_seq, stage=self.stage)

        # 拼接为 (num_stock, time, factor)
        data_X = torch.stack(train_X, dim=1)  # shape = [num_stock, lookback, factor_num]
        # print(rf"data_x的维度是：{data_X.shape}")
        if data_X.shape[0]<10:
            return['out_sample'],['out_sample'], ['out_sample'], ['out_sample'], ['out_sample']
        data_y = train_y 
        data_liquid = train_liquid
        data_time = label_date
        return data_X, data_y, data_time, code_value, data_liquid

    def __len__(self):
        return len(self.date_list) - self.lookback

    
    
# 【训练-验证-测试】的统一数据框架
class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list, test_date_list):
        super().__init__()
        self.args = args
        self.train = DLDataset(train_date_list, stage='train')
        self.val = DLDataset(valid_date_list, stage='valid')
        self.test = DLDataset(test_date_list, stage='test')
    
    def train_dataloader(self):
        # print("train_dataloader中的batch_size:", self.args.batch_size)
        return DataLoader(self.train, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          num_workers=2, shuffle=True, persistent_workers=False, drop_last=False, pin_memory=False)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn,
                          num_workers=0, persistent_workers=False, pin_memory=False, drop_last=False)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)