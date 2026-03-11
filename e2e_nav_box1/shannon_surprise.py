#!/usr/bin/env python3

import torch

class ShannonSurprise:
    def __init__(self, angles: torch.Tensor, num_bins: int, min_val: float = None, max_val: float = None):
        """
        全データのステアリング角度分布を元に、シャノンサプライズ計算法を初期化する。
        """
        self.num_bins = num_bins
        self.device = angles.device
        
        # 範囲が手動指定されていればそれを使い、なければデータの最小・最大を使う
        self.min_val = min_val if min_val is not None else angles.min().item()
        self.max_val = max_val if max_val is not None else angles.max().item()
        
        self.bins = torch.linspace(self.min_val, self.max_val, num_bins + 1, device=self.device)
        
        # 各角度がどのビンに属するかを数える（ヒストグラム）
        angles_binned = torch.bucketize(angles, self.bins) - 1
        angles_binned = angles_binned.clamp(min=0, max=num_bins - 1)
        
        # 確率 P(x) の計算
        bin_counts = torch.bincount(angles_binned, minlength=num_bins).float()
        self.bin_probs = bin_counts / (bin_counts.sum() + 1e-6)

    def compute_weights(self, angles: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        与えられた角度バッチに対して重み (-log P(x)) を計算する。
        """
        # 各角度をビンに割り当て
        angles_binned = torch.bucketize(angles, self.bins) - 1
        angles_binned = angles_binned.clamp(min=0, max=self.num_bins - 1)
        
        # サプライズ（重み）の計算
        return -torch.log(self.bin_probs[angles_binned] + eps)
