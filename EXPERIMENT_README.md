# 异常检测改进实验说明

本实验框架实现了三个针对点云异常检测的改进方案，并提供了系统化的实验配置。

## 三个改进方案

### 改进 1: Top-k 均值距离 (Top-k Mean Distance)
**原始逻辑**: 使用最小距离 `dist = np.min(distance_matrix, axis=1)`

**改进逻辑**: 找最近的 k 个距离，求平均
```python
topk_distances = torch.topk(dist, k=3, largest=False)
knn_dist = torch.mean(topk_distances, dim=1)
```

**可调参数**:
- `topk_k`: k值，默认3，推荐测试 [3, 5, 7]

**预期效果**: 减少单点噪声影响，提高鲁棒性

---

### 改进 2: 局部对比归一化 (Local Contrast Normalization, LCN)
**原始逻辑**: 计算出的 kNN 距离直接作为异常分数

**改进逻辑**: 分数除以空间邻居分数的中位数
```python
local_reference = torch.median(neighbor_scores, dim=1)
norm_scores = knn_dist / (local_reference + epsilon)
```

**可调参数**:
- `lcn_neighbors`: 邻居数量，默认5，推荐测试 [3, 5, 10]

**预期效果**: 抑制全局扫描偏差，凸显局部突变

---

### 改进 3: 对象级聚合优化 (Smooth Aggregation)
**原始逻辑**: `obj_score = np.max(norm_scores)` (只要一个点异常就报警)

**改进逻辑**: 
1. 空间平滑：对每个点用 k 近邻平均平滑
2. Top percentile 均值：取最高的 1% 点的均值

```python
# 空间平滑
smoothed_scores = torch.mean(neighbor_scores, dim=1)
# Top 1% 均值
num_top = int(len(smoothed_scores) * 0.01)
top_scores = torch.topk(smoothed_scores, k=num_top)
obj_score = torch.mean(top_scores)
```

**可调参数**:
- `smooth_k`: 平滑邻居数，默认10，推荐测试 [10, 15]
- `top_percentile`: 百分比，默认0.01 (1%)，推荐测试 [0.01, 0.02]

**预期效果**: 防止孤立飞点/噪声破坏 O-ROC 性能

---

## 实验配置

所有实验配置在 `experiment_configs.py` 中定义：

### 单一改进实验
- **baseline**: 原始方法
- **topk_k3, topk_k5, topk_k7**: 测试不同 k 值的 Top-k 均值
- **lcn_k3, lcn_k5, lcn_k10**: 测试不同邻居数的 LCN
- **smooth_k10_p1, smooth_k10_p2, smooth_k15_p1**: 测试不同参数的平滑聚合

### 组合改进实验
- **all_improvements**: 三个改进全部启用
- **topk_lcn**: Top-k + LCN
- **topk_smooth**: Top-k + 平滑聚合
- **lcn_smooth**: LCN + 平滑聚合

---

## 使用方法

### 1. 查看所有可用实验
```bash
python run_experiments.py --exp list
```

### 2. 运行单个实验
```bash
# 运行基线
python run_experiments.py --dataset minishift --class_name all --exp baseline

# 运行 Top-k=3 实验
python run_experiments.py --dataset minishift --class_name all --exp topk_k3

# 运行所有改进组合
python run_experiments.py --dataset minishift --class_name all --exp all_improvements
```

### 3. 运行所有实验（消耗时间较长）
```bash
python run_experiments.py --dataset minishift --class_name all --exp all
```

### 4. 针对特定类别测试
```bash
python run_experiments.py --dataset minishift --class_name <class_name> --exp all
```

---

## 实验输出

### 结果保存位置
结果保存在 `./experiment_results/` 目录：
- `results_<dataset>_<timestamp>.csv`: CSV格式结果
- `results_<dataset>_<timestamp>.json`: JSON格式结果

### 输出指标
- **Image ROC-AUC**: 图像级异常检测性能
- **Pixel ROC-AUC**: 像素级异常检测性能  
- **AU-PRO**: Area Under PRO curve (仅 MVTec3D)

### 实验报告包含
1. 每个实验在各类别上的详细结果
2. 各实验的平均性能对比
3. 最佳实验配置推荐

---

## 代码修改说明

### 新增文件
1. **feature_extractors/features_improved.py**: 改进版特征提取器
   - 包含三个改进方案的完整实现
   - 保留原始方法作为对照

2. **experiment_configs.py**: 实验配置文件
   - 定义了15+个实验配置
   - 方便添加新的参数组合

3. **run_experiments.py**: 实验运行脚本
   - 自动化运行多个实验
   - 自动保存和汇总结果

### 修改说明
- 原始代码保持不变
- 所有改进在 `features_improved.py` 中实现
- 通过参数开关控制是否启用改进

---

## 实验建议

### 第一阶段：单一改进效果评估
1. 运行 baseline
2. 分别运行三个改进的不同参数版本
3. 对比找出每个改进的最佳参数

推荐命令：
```bash
python run_experiments.py --dataset minishift --class_name all --exp baseline
python run_experiments.py --dataset minishift --class_name all --exp topk_k3
python run_experiments.py --dataset minishift --class_name all --exp topk_k5
python run_experiments.py --dataset minishift --class_name all --exp lcn_k5
python run_experiments.py --dataset minishift --class_name all --exp smooth_k10_p1
```

### 第二阶段：组合改进评估
使用第一阶段找到的最佳参数，测试组合效果：
```bash
python run_experiments.py --dataset minishift --class_name all --exp topk_lcn
python run_experiments.py --dataset minishift --class_name all --exp topk_smooth
python run_experiments.py --dataset minishift --class_name all --exp lcn_smooth
python run_experiments.py --dataset minishift --class_name all --exp all_improvements
```

### 第三阶段：深入分析
1. 分析哪些改进有效、哪些无效
2. 针对特定类别进行细粒度分析
3. 调整参数范围进行精细搜索

---

## 预期实验结果分析

### 改进1 (Top-k均值) 预期
- **可能有效**: 数据中存在噪声点时
- **可能无效**: 正常点和异常点距离分布差异明显时

### 改进2 (LCN) 预期  
- **可能有效**: 存在全局密度不均匀、局部对比度变化大的情况
- **可能无效**: 异常分布相对均匀时

### 改进3 (平滑聚合) 预期
- **可能有效**: 存在孤立噪声点导致误报的情况
- **可能无效**: 异常区域本身就很小、很局部时

---

## 参数调优建议

如果初始参数效果不理想，可以在 `experiment_configs.py` 中添加新配置：

```python
CUSTOM_EXPERIMENTS = [
    {
        'name': 'custom_exp',
        'use_topk_mean': True,
        'topk_k': 4,  # 尝试其他 k 值
        'use_lcn': True,
        'lcn_neighbors': 8,  # 尝试其他邻居数
        'use_smooth_aggregation': True,
        'smooth_k': 12,
        'top_percentile': 0.015,  # 尝试 1.5%
        'description': 'Custom parameter combination'
    }
]
```

然后将其添加到 `ALL_EXPERIMENTS` 列表中。

---

## 注意事项

1. **GPU内存**: 某些改进（特别是LCN）会增加内存使用，注意监控GPU显存
2. **计算时间**: 空间KNN计算会增加运行时间，特别是点云较大时
3. **参数敏感性**: 不同数据集的最佳参数可能不同，建议针对性调优
4. **结果稳定性**: 建议多次运行取平均，或固定随机种子

---

## 快速开始

```bash
# 1. 查看所有实验
python run_experiments.py --exp list

# 2. 运行快速测试（baseline vs 最佳组合）
python run_experiments.py --dataset minishift --class_name <one_class> --exp baseline
python run_experiments.py --dataset minishift --class_name <one_class> --exp all_improvements

# 3. 如果效果好，运行完整评估
python run_experiments.py --dataset minishift --class_name all --exp all
```

实验结果会自动保存在 `./experiment_results/` 目录，包含详细的性能对比和最佳配置推荐。
