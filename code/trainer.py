from models import FactorTSMixer  
from losses import get_loss_fn
from torchmetrics.regression import PearsonCorrCoef
import pytorch_lightning as pl
import copy
import torch
import pandas as pd
import numpy as np
import gc
from config import params,get_basic_name
import os
import shutil
from datetime import datetime
import logging
import pickle
from typing import List, Tuple, Optional
from dateutil.relativedelta import relativedelta
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor,
    ModelCheckpoint, StochasticWeightAveraging)
from data_preprocessing import DLDataModule
import threading
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 实时监控显存使用情况
# def realtime_gpu_monitor(interval=2):
"""
后台实时监控 GPU 显存使用，每 interval 秒输出一次。
"""
    # def monitor():
    #     while True:
    #         if torch.cuda.is_available():
    #             mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    #             max_mem_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    #             print(f"[GPU Monitor] 当前占用: {mem_allocated:.2f} MB, 峰值占用: {max_mem_allocated:.2f} MB")
    #         time.sleep(interval)

    # t = threading.Thread(target=monitor, daemon=True)  # daemon=True 保证主程序结束后自动退出
    # t.start()
    # return t

    # 添加每隔2s打印一次输出日志
# class TimeBasedProgressBar(TQDMProgressBar):
#     def __init__(self,refresh_interval_seconds = 2.0):
#         super().__init__()
#         self.refresh_interval_seconds = refresh_interval_seconds
#         self.last_refresh_time = time.time()
#     def _update_progress_bar(self, progress_bar):
#         """辅助函数，用于刷新指定的进度条"""
#         current_time = time.time()
#         if current_time - self.last_refresh_time >= self.refresh_interval_seconds:
#             progress_bar.update(0)
#             self.last_refresh_time = current_time

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self._update_progress_bar(self.train_progress_bar)
#         return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self._update_progress_bar(self.val_progress_bar)
#         return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)

#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self._update_progress_bar(self.test_progress_bar)
#         return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
class TimeBasedProgressBar(TQDMProgressBar):
    def __init__(self, refresh_interval_seconds=2.0):
        super().__init__()
        self.refresh_interval_seconds = refresh_interval_seconds
        self._stop_flags = {'train': False, 'val': False, 'test': False}
        self._threads = {'train': None, 'val': None, 'test': None}

    def _refresh_loop(self, phase, progress_bar_attr):
        while not self._stop_flags[phase]:
            progress_bar = getattr(self, progress_bar_attr, None)
            if progress_bar is not None:
                progress_bar.refresh()
                print(progress_bar.format_meter(
                    n=progress_bar.n,
                    total=progress_bar.total,
                    elapsed=progress_bar.format_dict['elapsed']
                ))
            time.sleep(self.refresh_interval_seconds)

    def _start_refresh_thread(self, phase, progress_bar_attr):
        self._stop_flags[phase] = False
        t = threading.Thread(target=self._refresh_loop, args=(phase, progress_bar_attr), daemon=True)
        self._threads[phase] = t
        t.start()

    def _stop_refresh_thread(self, phase):
        self._stop_flags[phase] = True
        t = self._threads.get(phase)
        if t is not None:
            t.join()

    # 覆盖 Lightning 回调函数：训练阶段
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self._start_refresh_thread('train', 'train_progress_bar')

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._stop_refresh_thread('train')

    # 验证阶段
    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self._start_refresh_thread('val', 'val_progress_bar')

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self._stop_refresh_thread('val')

    # 测试阶段
    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self._start_refresh_thread('test', 'test_progress_bar')

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self._stop_refresh_thread('test')
        
class DLLitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FactorTSMixer(args)  
        self.test_pearson = PearsonCorrCoef()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, *args):
        return self.model(*args)

    
#training_step函数易出错!原因：batch不是一个列表！而是将多个单变量拼接在一起，形成一个大张量！！
    def training_step(self, batch, batch_idx):
        # 假设 batch 是 (x, y, dates, codes, liquidity)
        # x, y, _, _, _ = batch
        

        tsdata, ret, time, code_value, liquid = batch  # 
        print(f"[Debug] ret mean: {ret.mean().item():.4f}, std: {ret.std().item():.4f}, min: {ret.min()}, max: {ret.max()}")
        loss_fn = get_loss_fn(self.args.loss)

        # 调整维度以适配 forward 和 loss
        ret = ret.unsqueeze(1)         # [N] -> [N, 1]
        preds = self.forward(tsdata)  # [N, 1]
        print(f"[Debug] preds mean: {preds.mean().item():.4f}, std: {preds.std().item():.4f}, min: {preds.min().item():.4f}, max: {preds.max().item():.4f}")

        loss = loss_fn(preds, ret)

        # 可选记录日志
        self.log('train_loss', loss.detach())

        return loss


    # @torch.no_grad()
    #原先expected to be in range of [-2, 1], but got 2报错的核心原因在此：加载数据时遍历了循环for i in range（tsdatas），这样只会传入一天的数据而非一个滑窗！ 导致  
    # 传入模型的实际数据为二维而非三维（包含lookback_window这一维    \
    def _evaluate_step(self, batch, batch_idx, stage):
        """
        每个 batch 是一天的 [N, T, F] 输入数据。
        """
        def _compute_excess_return(preds, rets, liquids, money=1e9):
            _, sort = preds.sort(dim=0, descending=True, stable=True)
            sort = sort.squeeze()
            total_hold = torch.tensor(0.0, device=preds.device)
            total_earned = torch.tensor(0.0, device=preds.device)
            for num, idx in enumerate(sort):
                if num >= 500:
                    break
                if (money - total_hold) < 1:
                    break
                hold_money = min(money - total_hold, liquids[idx])
                total_hold += hold_money
                total_earned += rets[idx] * hold_money
            top500_idx = sort[:500]
            top500_ret = rets[top500_idx]
            print(f"[Debug] top500 mean ret: {top500_ret.mean()}.item():.4f")
            return total_earned / money
        tsdata, ret, time, code_value, liquid = batch  # 每个 batch 是单一天市场快照

        if isinstance(tsdata, list) and tsdata[0] == 'out_sample':
            return None  # 跳过占位符样本

        preds = self(tsdata).squeeze(1)  # [N]
        preds = (preds - preds.mean()) / (preds.std() + 1e-6)
        loss_fn = get_loss_fn(self.args.loss)
        loss = loss_fn(preds, ret)

        # === 计算超额收益 ===
        excess_return = _compute_excess_return(preds, ret, liquid, money=1e9)

        #检查rets和liquids原始数据是否含杂质
        print(f"[Debug] liquids max: {liquid.max()}, min: {liquid.min()}, mean: {liquid.mean()}")
        print(f"[Debug] rets max: {ret.max()}, min: {ret.min()}, mean: {ret.mean()}")
        print(f"[Debug] excess_return: {excess_return}")

        # === 保存预测值（仅 test 阶段） ===
        if stage == "test":
            preds_cpu = preds.detach().cpu().numpy()
            res = pd.DataFrame({'Code': code_value, 'value': preds_cpu})
            res.index = res['Code']
            res.drop(columns=['Code'], inplace=True)
            # res['time'] = time
            res.to_pickle(f'{params.test_save_path}/{time}.pkl')

        # === 计算 IC ===
        ic_score = self.test_pearson(preds, ret)
        #检查preds和ret中是否含有nan
        print(f"[Debug] preds std: {preds.std()}, ret std: {ret.std()}, ic_score: {ic_score}")
        assert not torch.isnan(preds).any()
        assert not torch.isnan(ret).any()

        # === 缓存结果 ===
        if stage == 'val':
            self.validation_step_outputs.append([excess_return, ic_score])
        elif stage == "test":
            self.test_step_outputs.append([excess_return, ic_score])

        return {
            "loss": loss.detach().cpu(),
            "excess_return": excess_return,
            "ic": ic_score,
        }
        # tsdata, rets, times, code_values, liquids = batch
        # preds = self.forward(tsdata).squeeze(1).detach()

        # # 保存结果（只做测试集时）
        # if stage == "test":
        #     res = pd.DataFrame(preds.cpu().detach().numpy(), index=code_values, columns=['value'])
        #     res.index.name = 'Code'
        #     res['time'] = times
        #     res.to_pickle(f'{params.test_save_path}/step{batch_idx}.pkl')

        # excess_return = self._compute_excess_return(preds, rets, liquids)
        # ic = self.test_pearson(preds, rets)

        # if stage == 'val':
        #     self.validation_step_outputs.append([excess_return, ic])
        # elif stage == 'test':
        #     self.test_step_outputs.append([excess_return, ic])

        # del tsdata,rets,times,code_values,liquids,preds
        # torch.cuda.empty_cache()


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self._evaluate_step(batch, batch_idx, 'val')
            torch.cuda.empty_cache()  # 手动清除缓存
            return output

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self._evaluate_step(batch, batch_idx, 'test')
            torch.cuda.empty_cache()
            return output

    def on_train_epoch_end(self):
        super().on_train_end()
        torch.cuda.empty_cache()
        gc.collect()
    def on_validation_epoch_end(self):
        """
        计算验证集上 val_wei = excess_return * 50 + ic 的平均值。
        清空缓存，释放内存。
        """
        val_step_outputs = self.validation_step_outputs
        if not val_step_outputs:
            return

        val_scores = [
            (data[0] * 50 + data[1]) for data in val_step_outputs if data is not None
        ]
        mean_score = sum(val_scores) / len(val_scores)

        self.log('val_wei', mean_score, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()


    def on_test_epoch_end(self):
        """
        计算测试集上平均超额收益（excess return）。
        清空缓存，释放内存。
        """
        test_step_outputs = self.test_step_outputs
        if not test_step_outputs:
            return

        excess_returns = [
            data[0] for data in test_step_outputs if data is not None
        ]
        mean_return = sum(excess_returns) / len(excess_returns)

        self.log('test_wei', mean_return, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()


    def configure_optimizers(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = {'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
                     'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs)}[self.args.optimizer]
        return {'optimizer': optimizer}

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=10, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7, device='gpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_wei', mode='max', patience=20))
        callbacks.append(TimeBasedProgressBar())
        return callbacks



# 单次训练
def train_single(args, name, seed, train_date_list, valid_date_list, test_date_list):
    torch.set_num_threads(args.threads)
    seed_everything(seed)

    # ✅ 设置日志与性能分析器
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    profiler = SimpleProfiler(dirpath=params.profiler_path, filename=name)

    # ✅ 自动识别可用 trainer 参数（防止 PyTorch Lightning 报错）
    args_for_trainer = dict()
    for key, value in vars(args).items():
        try:
            Trainer(**{key: value})
            args_for_trainer[key] = value
        except:
            pass

    litmodel = DLLitModule(args)
    # ✅ 补充强烈推荐参数
    args_for_trainer.setdefault('accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
    args_for_trainer.setdefault('num_sanity_val_steps', 1)
    args_for_trainer.setdefault('logger', logger)
    args_for_trainer.setdefault('profiler', profiler)
    args_for_trainer.setdefault('deterministic', True)
    args_for_trainer.setdefault('callbacks',litmodel.configure_callbacks())
    #训练器改为AMP（自动混合精度）
    # args_for_trainer.update({
    #     "precision": 16,
    # })
    # 监控显存使用情况
    # gpu_monitor_thread = realtime_gpu_monitor(interval = 2)
    # ✅ 初始化训练器  
    # 并限制验证集的大小\
    trainer = Trainer(**args_for_trainer)

    # ✅ 初始化模型与数据模块
    
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    litmodel.save_hyperparameters(args)  # ✅ 保存配置（便于调参可视化）
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)

    # ✅ 训练 & 测试流程
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm)

    # ✅ 打印评估结果
    print("✅ 测试结果:", test_result)


# 训练主函数
def train(config, args, name, market, season, state='train'):
    # ========== 路径设置与备份 ==========
    save_path = f"{config.root_path}/model_test/{get_basic_name()}"    #model_test下属各模型文件夹
    if os.path.exists(save_path):
        print(f"[INFO] Removing previous save directory: {save_path}")
        shutil.rmtree(save_path)
    if os.path.exists('logs'):
        print("[INFO] Removing lightning_logs...")
        shutil.rmtree('logs')

    # 删除默认的 ckpt 保存路径
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    if os.path.exists(ckpt_dir):
        print("[INFO] Removing checkpoints...")
        shutil.rmtree(ckpt_dir)

    os.makedirs(save_path, exist_ok=True)
    # ==================================================

    try:
        # 备份模型脚本
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)
    # os.makedirs(save_path, exist_ok=True)
    # try:
    #     shutil.copy(os.path.abspath(__file__), save_path)
    # except Exception as e:
    #     print(f"备份当前脚本失败: {e}")

    # 模型保存路径配置
    params.model_name = f"{save_path}/{name[:len(market) + 19]}"          # ALL2023q1-validperiod1.pth
    params.test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}"
    os.makedirs(params.test_save_path, exist_ok=True)

    # ========== 因子文件读取 ==========
    sel_fac_name_save = [x for x in os.listdir(f'{config.fac_path}/{config.fac_name}') if season in x]
    assert len(sel_fac_name_save) == 1, "因子文件匹配错误"
    sel_fac_name_save = sel_fac_name_save[0]

    all_data = pd.read_feather(f'{config.fac_path}/{config.fac_name}/{sel_fac_name_save}')
    date_list = sorted([x for x in all_data["date"].unique()
                        if x in params.ret_data.index and x in params.liquid_data.index])

    all_data = all_data.set_index("date").sort_index()
    params.all_data = all_data

    # ===== 特征列识别 =====
    non_factor_cols = ['Code', 'Label', 'liquid']
    feature_map = [col for col in all_data.columns if col not in non_factor_cols]
    params.factor_list = feature_map
    params.factor_num = len(feature_map)

    with open(f'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(f'{factor_name}={idx}\n')
    # ========== 时间划分 ==========
    def get_train_date_split(season, period):
        # 从季节字符串中提取年份和季度，计算测试集的起始日期
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        # 将字符串转换为日期对象
        start_date = datetime.strptime(test_start, "%Y%m")  # start date即为测试集（为期一个季度，也就是3个月）开始时间，具体见main.py
        # 初始化一个空列表，用于存储有效的日期分割点
        valid_date_split = []
        for i in [-3, 0, 6, 12, 18, 24]:
            valid_date_split.append((start_date - relativedelta(months=i)).strftime("%Y%m"))
        valid_date_split.reverse()
        train_start = valid_date_split[0]        # 验证集时长为6个月，即上述valid_date_split中的中间4个元素相邻差，从前24个月开始就是，分别向前推6个月，共4个周期
        valid_start = valid_date_split[period - 1]
        valid_end = valid_date_split[period]
        test_end = valid_date_split[-1]
        return train_start, valid_start, valid_end, test_start, test_end

    train_start, valid_start, valid_end, test_start, test_end = get_train_date_split(season, period=int(params.test_save_path[-1]))
    train_start = max(train_start, "202101")

    lookback = params.lookback_window

    def build_rolling_dates(date_list, start, end):
        # 构造滑窗时间序列：从第 lookback 天开始，每天都可以形成一个窗口
        return [date_list[i] for i in range(len(date_list)) if
                date_list[i - lookback:i] and start <= date_list[i] < end]

    valid_date_list = build_rolling_dates(date_list, valid_start, valid_end)
    train_date_list = build_rolling_dates(date_list, train_start, valid_start) + \
                      build_rolling_dates(date_list, valid_end, test_start)
    test_date_list = build_rolling_dates(date_list, test_start, test_end)

    # 排除极端行情
    not_train_date = [x for x in date_list if "202402" <= x <= "20240223"]
    train_date_list = [x for x in train_date_list if x not in not_train_date]

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        # 过滤掉全为0的因子列
        params.all_data = params.all_data.loc[:, params.all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError("当前仅支持 market == 'ALL'")

    # ========== 模型结构参数设置（TSMixer用） ==========
    # 如需动态设置，也可写入 config 或 args
    if not hasattr(params, 'lookback_window'):
        params.lookback_window = 5
        params.hidden_size = 64
        params.mixer_layers = 2

    # ========== 启动训练 ==========
    if state == 'train':
        train_single(args, name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError("暂不支持非训练模式")
