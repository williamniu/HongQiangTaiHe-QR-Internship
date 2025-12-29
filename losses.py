import torch
import torch.nn.functional as F
#修改为以下损失函数后就可以跑通了！可是训练逻辑待考虑：  
# 下述损失函数建立在逐日计算损失的基础上，可我本来的训练逻辑应该是每一batch进行一次损失函数计算\
# debug整体感悟：把一个batch看作一个整体，而不是遍历每天！  因子特征、输出收益率和标签收益率都如此。

#原先损失函数1：（wpcc，排名加权相关系数）
# def wpcc(preds, y):
#     """
#     加权 Pearson Correlation Coefficient 损失函数（越大越好，因此最终返回负值）

#     Args:
#         preds: Tensor, shape = [B*N] 或 [B*N, 1]，模型预测收益
#         y: Tensor, shape = [B*N] 或 [B*N, 1]，真实收益标签

#     Returns:
#         Scalar Tensor: 加权皮尔逊损失（负值）
#     """
#     # ----------- Step 1: 统一预测值和标签形状 [B*N, 1] -----------
#     preds = preds.view(-1, 1).float()
#     y = y.view(-1, 1).float()

#     # ----------- Step 2: 排序并生成指数加权因子 -----------
#     _, argsort = torch.sort(preds, descending=True, dim=0)  # 获取排名索引
#     weight = torch.zeros_like(preds)                        # 初始化权重 [B*N, 1]
#     weight_new = torch.tensor(
#         [0.5 ** ((i - 1) / (preds.shape[0] - 1)) for i in range(1, preds.shape[0] + 1)],
#         dtype = preds.dtype,
#         device=preds.device
#     ).unsqueeze(1)  # [B*N, 1]

#     weight[argsort, 0] = weight_new.squeeze(1)  # 按排序赋权

#     # ----------- Step 3: 加权协方差计算（公式展开）-----------
#     weighted_mean_pred = (preds * weight).sum(dim=0) / weight.sum(dim=0)
#     weighted_mean_y = (y * weight).sum(dim=0) / weight.sum(dim=0)
#     #以下公式计算协方差：是由定义式展开后化简后的结果
#     wcov = (preds * y * weight).sum(dim=0) / weight.sum(dim=0) - \
#            weighted_mean_pred * weighted_mean_y

#     pred_std = torch.sqrt(((preds - weighted_mean_pred) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))
#     y_std = torch.sqrt(((y - weighted_mean_y) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))

# # Ensure the standard deviations are not too small (thresholding)
#     pred_std= torch.max(pred_std, torch.tensor(1e-6, device=pred_std.device))  # 限制最小值
#     y_std = torch.max(y_std, torch.tensor(1e-6, device=y_std.device))  # 限制最小值

#     w_pcc = wcov / (pred_std * y_std)  # 不再加 small constant，因为我们已经确保标准差不为零

#     return -w_pcc.mean()  # 损失函数为负的 weighted Pearson

#损失函数2：soft_rank损失函数，使排序可导，但可能导致梯度过于“平滑”，选股效果较弱
def soft_rank(x, temperature=0.1, eps=1e-6):
    """
    Soft-rank function to approximate ranks in a differentiable way.
    Input:
        x: Tensor of shape [N, 1]
        temperature: Softness parameter (smaller -> closer to hard ranking)
    Output:
        Tensor of shape [N, 1] representing soft ranks
    """
    x = x.view(-1, 1)
    pairwise_diff = x - x.T  # [N, N]
    soft = torch.sigmoid(-pairwise_diff / temperature)  # sigmoid trick
    return soft.sum(dim=-1, keepdim=True) + 0.5  # Soft rank in [1, N]

def wpcc_soft(preds, y, temperature=0.1, alpha=1e-4):
    """
    Differentiable Weighted Pearson Correlation Loss with Soft-Rank and L2 regularization.

    Args:
        preds: [B*N] or [B*N, 1]
        y:     [B*N] or [B*N, 1]
        temperature: softness of the ranking
        alpha: weight for L2 regularization on predictions

    Returns:
        Scalar Tensor (loss to minimize)
    """
    preds = preds.view(-1, 1).float()
    y = y.view(-1, 1).float()

    # -------- Soft Rank + Exponential Weighting --------
    soft_ranks = soft_rank(preds, temperature=temperature)  # [B*N, 1]

    # 生成指数衰减权重（权重越靠前越大）
    N = preds.shape[0]
    weight_base = torch.tensor(
        [0.5 ** ((i - 1) / (N - 1)) for i in range(1, N + 1)],
        dtype=preds.dtype,
        device=preds.device
    ).unsqueeze(1)  # [N, 1]

    # 根据 soft-rank 位置分配权重（排序靠前 → 权重大）
    sorted_weights = weight_base
    sorted_indices = torch.argsort(soft_ranks.squeeze(), dim=0, descending=False)
    weights = torch.zeros_like(preds)
    weights[sorted_indices] = sorted_weights

    # -------- 加权 Pearson 相关计算 --------
    mean_pred = (preds * weights).sum() / weights.sum()
    mean_y = (y * weights).sum() / weights.sum()

    wcov = ((preds * y * weights).sum() / weights.sum()) - mean_pred * mean_y
    std_pred = torch.sqrt(((preds - mean_pred) ** 2 * weights).sum() / weights.sum())
    std_y = torch.sqrt(((y - mean_y) ** 2 * weights).sum() / weights.sum())

    std_pred = torch.clamp(std_pred, min=1e-6)
    std_y = torch.clamp(std_y, min=1e-6)

    wpcc = wcov / (std_pred * std_y)

    # -------- 添加正则项（惩罚过大预测） --------
    l2_penalty = (preds ** 2).mean()

    return -wpcc.mean() + alpha * l2_penalty

# def neg_pearson_loss(preds, y):
#     preds = preds.view(-1).float()
#     y = y.view(-1).float()

#     xm = preds - preds.mean()
#     ym = y - y.mean()

#     r_num = (xm * ym).sum()
#     r_den = torch.sqrt((xm ** 2).sum() + 1e-6) * torch.sqrt((ym ** 2).sum() + 1e-6)

#     r = r_num / r_den
#     return -r

# ========= 包装接口 =========
def get_loss_fn(loss):
    """
    损失函数获取接口
    """
    loss_dict = {
        'wpcc': wpcc_soft,
        # 可扩展其他损失函数，如 'mse': nn.MSELoss()
    }
    return loss_dict[loss]
