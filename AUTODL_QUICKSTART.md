# Autodl 服务器实验快速启动指南

## 环境信息
- **数据集位置**: `/root/autodl-fs/Real3D-AD-PCD/Real3D-AD-PCD`
- **权重位置**: `/root/autodl-fs/output_real3dad/output_real3dad/snapshots/snapshot.pth.tar`
- **结果保存**: `/root/autodl-fs/experiment_results`
- **GPU**: 4090

## 快速开始

### 1. 上传代码到服务器
将以下文件上传到autodl服务器的项目目录：
```
MiniShift-Simple3D-main/
├── feature_extractors/
│   ├── features.py (原始)
│   ├── features_improved.py (改进版)
│   └── FPFH.py
├── experiment_configs.py
├── run_autodl_experiments.py
└── ...其他原有文件
```

### 2. 查看所有可用实验
```bash
python run_autodl_experiments.py --exp list
```

### 3. 运行基线实验（推荐先运行）
```bash
# 测试单个类别
python run_autodl_experiments.py --exp baseline --class_name airplane

# 测试所有类别
python run_autodl_experiments.py --exp baseline --class_name all
```

### 4. 运行改进实验

#### 测试改进1: Top-k均值距离
```bash
python run_autodl_experiments.py --exp topk_k3 --class_name all
python run_autodl_experiments.py --exp topk_k5 --class_name all
python run_autodl_experiments.py --exp topk_k7 --class_name all
```

#### 测试改进2: 局部对比归一化
```bash
python run_autodl_experiments.py --exp lcn_k5 --class_name all
python run_autodl_experiments.py --exp lcn_k10 --class_name all
```

#### 测试改进3: 平滑聚合
```bash
python run_autodl_experiments.py --exp smooth_k10_p1 --class_name all
python run_autodl_experiments.py --exp smooth_k10_p2 --class_name all
```

#### 测试组合改进
```bash
python run_autodl_experiments.py --exp all_improvements --class_name all
```

### 5. 运行所有实验（时间较长，建议使用screen）
```bash
# 启动screen会话
screen -S experiments

# 运行所有实验
python run_autodl_experiments.py --exp all --class_name all

# 按 Ctrl+A+D 退出screen，程序继续运行
# 重新连接: screen -r experiments
```

---

## 实验建议流程

### 阶段1: 快速验证（约1-2小时）
先在单个类别上测试所有配置，快速识别有效改进：
```bash
python run_autodl_experiments.py --exp baseline --class_name airplane
python run_autodl_experiments.py --exp topk_k3 --class_name airplane
python run_autodl_experiments.py --exp lcn_k5 --class_name airplane
python run_autodl_experiments.py --exp smooth_k10_p1 --class_name airplane
python run_autodl_experiments.py --exp all_improvements --class_name airplane
```

### 阶段2: 参数优化（约2-4小时）
针对有效的改进，测试不同参数：
```bash
# 如果Top-k有效，测试不同k值
python run_autodl_experiments.py --exp topk_k3 --class_name all
python run_autodl_experiments.py --exp topk_k5 --class_name all

# 如果LCN有效，测试不同邻居数
python run_autodl_experiments.py --exp lcn_k3 --class_name all
python run_autodl_experiments.py --exp lcn_k5 --class_name all
```

### 阶段3: 完整评估（约6-12小时）
在所有类别上运行最佳配置：
```bash
python run_autodl_experiments.py --exp all --class_name all
```

---

## 结果查看

### 实时查看结果
```bash
# 查看最新CSV结果
cat /root/autodl-fs/experiment_results/results_real3d_latest.csv

# 使用pandas分析
python -c "
import pandas as pd
df = pd.read_csv('/root/autodl-fs/experiment_results/results_real3d_latest.csv')
print(df.groupby('experiment')['image_rocauc'].mean().sort_values(ascending=False))
"
```

### 下载结果
```bash
# 从autodl下载结果文件
# 在本地执行：
scp root@connect.westb.seetacloud.com:/root/autodl-fs/experiment_results/*.csv ./
```

---

## 后台运行（推荐）

使用nohup在后台运行，避免ssh断开导致实验中断：

```bash
# 后台运行所有实验
nohup python run_autodl_experiments.py --exp all --class_name all > experiment.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看实时日志
tail -f experiment.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

---

## 实验配置说明

### 15个预定义实验

| 实验名称 | 改进方案 | 参数 | 说明 |
|---------|---------|------|------|
| `baseline` | 无 | - | 原始方法 |
| `topk_k3` | Top-k均值 | k=3 | 用3个最近距离的均值 |
| `topk_k5` | Top-k均值 | k=5 | 用5个最近距离的均值 |
| `topk_k7` | Top-k均值 | k=7 | 用7个最近距离的均值 |
| `lcn_k3` | LCN | neighbors=3 | 3邻居局部归一化 |
| `lcn_k5` | LCN | neighbors=5 | 5邻居局部归一化 |
| `lcn_k10` | LCN | neighbors=10 | 10邻居局部归一化 |
| `smooth_k10_p1` | 平滑聚合 | k=10, p=1% | 10邻居平滑+Top1% |
| `smooth_k10_p2` | 平滑聚合 | k=10, p=2% | 10邻居平滑+Top2% |
| `smooth_k15_p1` | 平滑聚合 | k=15, p=1% | 15邻居平滑+Top1% |
| `topk_lcn` | 组合 | - | Top-k + LCN |
| `topk_smooth` | 组合 | - | Top-k + 平滑聚合 |
| `lcn_smooth` | 组合 | - | LCN + 平滑聚合 |
| `all_improvements` | 组合 | - | 三个改进全开 |

---

## 预期运行时间

基于Real3D数据集（12个类别）：

- **单个实验，单个类别**: ~5-10分钟
- **单个实验，所有类别**: ~1-2小时
- **所有实验（15个），所有类别**: ~15-30小时

实际时间取决于：
- 点云大小
- GPU性能（4090应该较快）
- 是否启用改进（LCN会增加计算时间）

---

## 故障排除

### 1. CUDA内存不足
```python
# 修改 run_autodl_experiments.py 中的参数
--num_group 128  # 减少group数量（默认256）
--group_size 32  # 减少group大小（默认64）
```

### 2. 数据集路径错误
检查 `data/real3d.py` 中的 `DATASETS_PATH` 是否正确：
```python
DATASETS_PATH = '/root/autodl-fs/Real3D-AD-PCD/Real3D-AD-PCD'
```

### 3. 导入错误
确保所有依赖已安装：
```bash
pip install torch open3d scikit-learn pandas tqdm
```

### 4. 查看详细错误
```bash
# 使用verbose模式
python run_autodl_experiments.py --exp baseline --class_name airplane 2>&1 | tee debug.log
```

---

## 自定义实验

如果需要测试其他参数组合，编辑 `experiment_configs.py`：

```python
CUSTOM_EXPERIMENTS = [
    {
        'name': 'my_custom_exp',
        'use_topk_mean': True,
        'topk_k': 4,  # 自定义k值
        'use_lcn': True,
        'lcn_neighbors': 7,  # 自定义邻居数
        'use_smooth_aggregation': False,
        'description': 'Custom parameters test'
    }
]

# 添加到 ALL_EXPERIMENTS
ALL_EXPERIMENTS = [BASELINE] + TOPK_EXPERIMENTS + ... + CUSTOM_EXPERIMENTS
```

然后运行：
```bash
python run_autodl_experiments.py --exp my_custom_exp --class_name all
```

---

## 结果分析示例

实验完成后，使用Python分析结果：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('/root/autodl-fs/experiment_results/results_real3d_latest.csv')

# 按实验分组
summary = df.groupby('experiment').agg({
    'image_rocauc': ['mean', 'std'],
    'pixel_rocauc': ['mean', 'std']
}).round(3)

print(summary)

# 找出最佳实验
best = df.groupby('experiment')['image_rocauc'].mean().sort_values(ascending=False)
print("\n最佳实验排名:")
print(best.head(5))

# 对比baseline
baseline_score = df[df['experiment']=='baseline']['image_rocauc'].mean()
for exp in df['experiment'].unique():
    exp_score = df[df['experiment']==exp]['image_rocauc'].mean()
    improvement = (exp_score - baseline_score) * 100
    print(f"{exp}: {improvement:+.2f}%")
```

---

## 论文结果整理

实验完成后，整理表格用于论文：

```python
import pandas as pd

df = pd.read_csv('/root/autodl-fs/experiment_results/results_real3d_latest.csv')

# 创建论文表格
table = df.pivot_table(
    index='class_name',
    columns='experiment',
    values='image_rocauc',
    aggfunc='mean'
).round(3)

# 添加平均行
table.loc['Average'] = table.mean()

# 保存为LaTeX格式
table.to_latex('/root/autodl-fs/experiment_results/paper_table.tex')
print(table)
```

---

## 联系与支持

如有问题：
1. 查看 `EXPERIMENT_README.md` 了解详细原理
2. 查看 `experiment.log` 了解运行日志
3. 检查 GPU 使用情况: `nvidia-smi`
4. 检查磁盘空间: `df -h`

实验愉快！🚀
