# MentalHealth-GraphLLM

基于**图神经网络**和**GraphRAG + LLM**的多模态心理健康预测系统。

## 研究目标

分析被动传感数据与心理问卷之间的关系，生成可解释的临床报告。

**核心问题**:
- 人格特质 + 手机行为 → 能否检测抑郁？
- 数字生物标志物 → 对学业表现有什么影响？

## 研究框架

```
Phase 1: 关系发现          Phase 2: 关系验证
┌─────────────────┐       ┌─────────────────┐
│ 传感器数据      │       │ PC算法因果发现   │
│       +        │ ──→   │       +         │
│ 问卷数据       │       │ Granger因果检验  │
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
  [统计关联分析]            [已验证关系图谱]
  [多重比较校正]
  [效应量筛选]

Phase 3: GNN预测           Phase 4: 报告生成
┌─────────────────┐       ┌─────────────────┐
│ 知识图谱        │       │ GraphRAG检索    │
│       +        │ ──→   │       +         │
│ 时序GAT编码    │       │ LLM解释生成     │
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
  [用户风险预测]            [临床可解释报告]
```

## 项目结构

```
mentalhealth-graphllm/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   └── datasets/              # 数据集配置
│       └── studentlife.yaml
│
├── src/
│   ├── data/                  # 数据加载
│   │   └── loaders/
│   │       ├── base.py        # 抽象基类（可扩展）
│   │       └── studentlife.py # StudentLife加载器
│   │
│   ├── analysis/              # 统计分析
│   │   ├── discovery/
│   │   │   └── correlation.py # 相关性 + 多重比较
│   │   └── validation/
│   │       └── causal.py      # PC算法 + Granger因果
│   │
│   ├── knowledge/             # 知识图谱
│   │   └── schemas/
│   │       └── dsm5.py        # DSM-5症状定义
│   │
│   ├── models/                # 模型
│   │   └── gnn/
│   │       └── encoders.py    # TemporalGAT
│   │
│   └── graphrag/              # 报告生成
│       ├── llm_client.py      # DeepSeek/Qwen客户端
│       └── explainer.py       # 报告生成器
│
├── scripts/                   # 执行脚本
│   └── 01_preprocess.py       # 数据预处理
│
├── data/raw/dataset/          # StudentLife原始数据
└── outputs/                   # 输出目录
```

## 快速开始

### 环境设置

```bash
# 使用已有的mamba环境
mamba activate qbio

# 或创建新环境
mamba create -n mentalhealth python=3.11
mamba activate mentalhealth
mamba install pytorch pytorch-geometric polars scipy -c pytorch -c pyg
pip install statsmodels pingouin causal-learn openai loguru omegaconf
```

### 运行预处理

```bash
python scripts/01_preprocess.py \
    --data-dir data/raw/dataset \
    --output outputs/processed
```

### 后续步骤

```bash
# 2. 关系发现
python scripts/02_discover_relations.py \
    --features outputs/processed/user_features.parquet \
    --labels outputs/processed/phq9_labels.parquet

# 3. 关系验证
python scripts/03_validate_relations.py

# 4. 构建知识图谱
python scripts/04_build_kg.py

# 5. 训练GNN (用户级特征)
python scripts/05_train_gnn.py \
    --features outputs/processed/user_features.parquet \
    --labels outputs/processed/phq9_labels.parquet

# 5. 训练GNN (时序特征 - 推荐)
python scripts/05_train_gnn.py \
    --features outputs/processed/temporal_features.parquet \
    --labels outputs/processed/phq9_labels.parquet \
    --use-temporal

# 6. 生成报告
python scripts/06_generate_reports.py --user_id u23
```

## 数据集

**StudentLife** (Dartmouth College)
- 参与者: ~51人
- 时长: 10周
- 传感器: GPS, 活动, 手机使用, 对话, WiFi, 蓝牙
- 问卷: PHQ-9, Big Five, PANAS, PSQI

## 核心模块

### 1. 统计分析 (`src/analysis/`)

```python
from src.analysis.discovery.correlation import RelationshipDiscovery

# 相关性分析 + FDR校正
discovery = RelationshipDiscovery()
results = discovery.compute_all_correlations(
    sensor_features, survey_scores,
    feature_names, survey_names
)

# 筛选显著且有意义的关系
significant = discovery.filter_significant(
    alpha=0.05,
    min_effect_size=0.3  # Cohen's d
)
```

### 2. 因果验证 (`src/analysis/validation/`)

```python
from src.analysis.validation.causal import CausalValidator

validator = CausalValidator()

# Granger因果检验
result = validator.granger_causality(
    time_series_x, time_series_y,
    max_lag=5
)

# PC算法因果发现
causal_graph = validator.pc_algorithm(data, variable_names)
```

### 3. GNN模型 (`src/models/gnn/`)

```python
from src.models.gnn.encoders import MentalHealthGNN

model = MentalHealthGNN(
    node_features=32,
    hidden_dim=64,
    embed_dim=32,
    n_heads=4,
    dropout=0.3
)

# 多任务预测
outputs = model(data_dict)
# outputs: phq9_score, bigfive, gpa, modality_attention
```

### 4. 报告生成 (`src/graphrag/`)

```python
from src.graphrag.llm_client import DeepSeekClient
from src.graphrag.explainer import ClinicalReportGenerator

llm = DeepSeekClient(api_key="...")
generator = ClinicalReportGenerator(llm)

report = generator.generate_report(user_profile, knowledge_context)
```

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 数据处理 | Polars | 比Pandas快3-10x |
| GNN | GAT | 注意力权重可解释 |
| 因果发现 | causal-learn | PC算法实现 |
| LLM | DeepSeek → Qwen | 先测试后本地化 |

## 严谨性保障

- **多重比较校正**: FDR/Bonferroni
- **效应量报告**: Cohen's d, 95% CI
- **因果vs相关**: 明确区分
- **可重复性**: 随机种子固定(42)

## 实验结果

### 数据概况

| 模态 | 每日记录数 | 用户数 |
|------|-----------|--------|
| GPS | 2,876 | 49 |
| Phone | 3,028 | 49 |
| Activity | 2,873 | 49 |
| Conversation | 2,721 | 49 |

特征维度:
- 用户级特征: (49, 39) - 聚合后的静态特征
- 时序特征: (3037, 257) - 滑动窗口特征 (1d, 3d, 7d)

### GNN模型性能 (5-fold CV, PHQ-9预测)

| 配置 | MSE | MAE | R² |
|------|-----|-----|-----|
| 用户级 (2 mod: GPS, Phone) | 27.70 ± 16.98 | 3.66 ± 1.18 | -0.51 ± 0.51 |
| 时序 (2 mod: GPS, Phone) | 25.61 ± 10.36 | 3.55 ± 0.86 | -0.25 ± 0.27 |
| 时序 (4 mod: 全部) | 24.25 ± 13.58 | 3.30 ± 0.86 | -0.27 ± 0.31 |

**结果分析**:
- 负R²表明模型性能低于均值基线
- 主要挑战: 样本量极小 (n=46-49) vs 高维特征
- 时序特征略优于用户级聚合特征
- 4模态优于2模态，但改善有限

### 下一步改进方向

1. **简化基线**: 尝试线性回归、Ridge、随机森林
2. **特征选择**: 使用LASSO或互信息筛选关键特征
3. **降维**: PCA/UMAP减少特征维度
4. **正则化**: 增强dropout、L2正则化
5. **数据增强**: 合成过采样或迁移学习

## 局限性

- 样本量小 (n≈50)
- 仅大学生群体
- 相关性研究，非因果证明
- **仅供研究使用，非临床诊断工具**

## 扩展接口

### 添加新数据集

```python
from src.data.loaders.base import BaseDatasetLoader

class GLOBEMLoader(BaseDatasetLoader):
    def load_surveys(self):
        # 实现
        pass
```

## 引用

```bibtex
@software{mentalhealth_graphllm,
  title = {MentalHealth-GraphLLM},
  year = {2025}
}
```

## License

Research use only.
