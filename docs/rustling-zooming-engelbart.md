# 多模态抑郁检测：从Baseline到高阶深度学习方法的实现计划

## 项目背景

**当前状态：**
- ✅ Phase 1: 数据预处理完成（46个用户，4个阳性样本）
- ✅ Phase 2: 特征工程完成（44个特征从GPS、App、通信、活动4个模态）
- ❌ Phase 3-4: Baseline建模未开始
- ❌ Phase 5-6: 高阶方法和可解释性未开始

**用户需求：**
1. 保留传统ML baseline（逻辑回归、随机森林、XGBoost）作为对比
2. 添加4种高阶深度学习方法：VAE、GNN、对比学习、Transformer
3. 重点关注特征学习与表示（而非仅分类性能）
4. 使用MacBook MPS加速（mamba环境：`qbio`）

**核心挑战：**
- 严重类别不平衡：4个阳性 vs 42个阴性（10.5:1）
- 小样本量（n=46）容易导致过拟合
- 需要可解释性来发现数字生物标记

---

## 实施路线图（7个阶段）

### Phase 3: Baseline模型（传统ML）

**目标：** 建立性能基准，验证特征有效性

#### 3.1 创建模型基础设施

**新建文件：**

1. **`src/models/baseline.py`** - Baseline模型封装
   ```python
   class BaselineModel:
       - LogisticRegression (L2正则化, balanced class weights)
       - RandomForest (max_depth=3, n_estimators=500, balanced)
       - XGBoost (scale_pos_weight=10.5 for imbalance)
   ```

2. **`src/models/evaluation.py`** - 评估指标和交叉验证
   ```python
   - stratified_cv(): 5折分层交叉验证
   - evaluate_model(): AUC-ROC, PR-AUC, sensitivity, specificity, F1
   - permutation_test(): 统计显著性检验（p<0.05）
   - plot_roc_curve(), plot_confusion_matrix()
   ```

3. **`src/utils/data_loader.py`** - 数据加载工具
   ```python
   - load_features_labels(): 加载combined_features.parquet + item9_labels_pre.csv
   - train_test_split_stratified(): 保持类别比例
   - apply_feature_scaling(): StandardScaler for continuous features
   ```

4. **`scripts/07_train_baseline.py`** - 训练脚本
   ```bash
   # 使用方式
   mamba activate qbio
   python scripts/07_train_baseline.py
   ```

5. **`notebooks/03_baseline_modeling.ipynb`** - 交互式分析

**输出：**
- `results/models/logistic_baseline.pkl`
- `results/models/random_forest_baseline.pkl`
- `results/models/xgboost_baseline.pkl`
- `results/metrics/baseline_metrics.json`
- `results/figures/baseline_roc_curves.png`
- `results/figures/feature_importance_comparison.png`

**预期性能：**
- AUC-ROC: 0.60-0.70（基于文献Saeb et al. 2015）
- Sensitivity ≥ 0.80（临床优先级：不遗漏阳性案例）
- 识别top 5-10个预测特征

---

### Phase 4: 深度学习基础设施

**目标：** 配置PyTorch + MPS，建立训练框架

#### 4.1 环境配置

**更新 `requirements.txt`：**
```txt
# 取消注释并更新PyTorch版本（支持Apple Silicon MPS）
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0  # For GNN
pytorch-metric-learning>=2.0.0  # For contrastive learning
```

**安装命令：**
```bash
mamba activate qbio
mamba install pytorch torchvision -c pytorch  # 自动启用MPS
pip install torch-geometric pytorch-metric-learning
```

#### 4.2 PyTorch训练基础设施

**新建文件：**

1. **`src/models/pytorch_base.py`** - PyTorch基类
   ```python
   class BaseDeepModel(nn.Module):
       - MPS device配置: device = "mps" if torch.backends.mps.is_available()
       - 通用训练循环: train(), validate(), test()
       - Early stopping (patience=20)
       - Model checkpointing (保存最佳模型)
       - Reproducibility (torch.manual_seed(42))
   ```

2. **`src/utils/pytorch_utils.py`** - PyTorch工具
   ```python
   - TabularDataset(Dataset): 包装44维特征 + 标签
   - get_dataloaders(): 创建train/val/test DataLoader
   - set_seed(): 固定随机种子
   - count_parameters(): 计算模型参数量
   ```

3. **`configs/model_configs.yaml`** - 超参数配置（新建文件）
   ```yaml
   common:
     batch_size: 16  # 小批量避免过拟合
     learning_rate: 0.001
     weight_decay: 0.01  # L2正则化
     max_epochs: 200
     early_stopping_patience: 20
     random_seed: 42

   vae:
     latent_dim: 8
     hidden_dims: [32, 16]
     beta: 1.0  # KL权重

   gnn:
     hidden_channels: 16
     num_layers: 2
     dropout: 0.3
     k_neighbors: 5  # KNN构图

   contrastive:
     temperature: 0.5
     projection_dim: 32
     augmentation_strength: 0.2

   transformer:
     d_model: 16
     nhead: 4
     num_layers: 2
     dropout: 0.2
   ```

---

### Phase 5A: Variational Autoencoder (VAE)

**目标：** 学习44维特征的低维潜在表示，用于异常检测和数据增强

#### 5A.1 模型架构

**新建文件：`src/models/vae_model.py`**

```python
class MultimodalVAE(BaseDeepModel):
    def __init__(self, input_dim=44, latent_dim=8, hidden_dims=[32, 16]):
        """
        Encoder: 44 -> 32 -> 16 -> latent_dim*2 (mean + logvar)
        Decoder: latent_dim -> 16 -> 32 -> 44
        """
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(44, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 44)
        )

    def loss_function(self, recon_x, x, mean, logvar, beta=1.0):
        """VAE Loss = Reconstruction + beta * KL divergence"""
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return MSE + beta * KLD
```

#### 5A.2 应用场景

**1. 异常检测（Anomaly Detection）**
- 计算重构误差：`||x - decoder(encoder(x))||^2`
- 假设：抑郁症患者（阳性样本）有更高的重构误差
- 阈值分类：重构误差 > threshold → 预测为阳性

**2. 数据增强（解决类别不平衡）**
- 从4个阳性样本的潜在分布中采样生成合成样本
- `z ~ N(mean_positive, var_positive)`
- `x_synthetic = decoder(z)`
- 生成20-30个合成阳性样本用于训练baseline

**3. 可视化潜在空间**
- t-SNE/UMAP可视化8维潜在向量
- 检查阳性/阴性样本是否在潜在空间中可分

**新建文件：**
- `scripts/08_train_vae.py` - VAE训练脚本
- `notebooks/04_vae_analysis.ipynb` - 潜在空间可视化

**输出：**
- `results/models/vae_best.pth`
- `results/figures/vae_latent_space.png` (t-SNE)
- `results/figures/vae_reconstruction_error.png`
- `data/processed/features/vae_synthetic_samples.parquet` (增强数据)

---

### Phase 5B: Graph Neural Network (GNN)

**目标：** 利用用户相似性图结构进行半监督学习

#### 5B.1 图构建策略

**新建文件：`src/features/graph_builder.py`**

```python
class UserSimilarityGraph:
    def build_knn_graph(features, k=5, metric='cosine'):
        """
        基于特征相似度构建K近邻图
        - 节点：46个用户
        - 边：连接k个最相似用户（基于44维特征余弦相似度）
        - 边权重：相似度分数
        """
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(features, k, metric=metric, include_self=False)
        return A  # 邻接矩阵
```

**图统计：**
- 节点数：46
- 边数：约 46 * 5 = 230（k=5）
- 节点特征：44维原始特征
- 节点标签：4个阳性（有监督）+ 42个阴性

#### 5B.2 GNN架构

**新建文件：`src/models/gnn_model.py`**

```python
from torch_geometric.nn import GCNConv, GATConv

class DepGraphNet(BaseDeepModel):
    """Graph Attention Network for depression prediction"""

    def __init__(self, in_channels=44, hidden_channels=16, num_layers=2):
        # 使用GAT而非GCN，因为attention可解释性更强
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.conv2 = GATConv(hidden_channels*4, hidden_channels, heads=1, dropout=0.3)
        self.classifier = nn.Linear(hidden_channels, 2)  # 二分类

    def forward(self, x, edge_index):
        # x: [46, 44] 节点特征
        # edge_index: [2, num_edges] 边连接
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        out = self.classifier(x)
        return out, x  # 返回logits和节点嵌入
```

#### 5B.3 训练策略

**半监督学习：**
- 使用所有46个节点的特征
- 仅用4个阳性 + 部分阴性样本的标签训练（模拟标注成本）
- 图卷积传播监督信号到未标注节点

**交叉验证：**
- Leave-One-Out CV（n=46太小不适合k-fold）
- 每次留一个节点作为测试，其余训练

**新建文件：**
- `scripts/09_train_gnn.py`
- `notebooks/05_gnn_analysis.ipynb` - 注意力权重可视化

**输出：**
- `results/models/gnn_best.pth`
- `results/figures/gnn_attention_weights.png` (哪些用户连接重要)
- `results/figures/gnn_node_embeddings.png` (t-SNE可视化)

---

### Phase 5C: 对比学习（Contrastive Learning）

**目标：** 在小样本场景下学习判别性表示

#### 5C.1 数据增强策略（Tabular Data）

**新建文件：`src/utils/augmentation.py`**

```python
class TabularAugmentation:
    """表格数据增强方法"""

    @staticmethod
    def mixup(x1, x2, alpha=0.2):
        """Mixup: 线性插值两个样本"""
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2

    @staticmethod
    def gaussian_noise(x, std=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * std
        return x + noise

    @staticmethod
    def feature_cutout(x, p=0.2):
        """随机遮盖部分特征（类似Dropout）"""
        mask = torch.rand(x.shape) > p
        return x * mask
```

#### 5C.2 对比学习框架

**新建文件：`src/models/contrastive_model.py`**

```python
from pytorch_metric_learning import losses

class ContrastiveEncoder(BaseDeepModel):
    """SimCLR-style contrastive learning"""

    def __init__(self, input_dim=44, projection_dim=32):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(44, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        # Projection head (for contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(32, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # 表示向量
        z = self.projector(h)  # 投影向量（用于对比损失）
        return h, z
```

**损失函数：NT-Xent (SimCLR)**
```python
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    z_i, z_j: 同一样本的两个增强视图
    拉近正样本对，推开负样本对
    """
    from pytorch_metric_learning.losses import NTXentLoss
    loss_fn = NTXentLoss(temperature=temperature)
    return loss_fn(z_i, z_j, labels)
```

#### 5C.3 训练策略

**正样本对构建：**
- 同一用户的两次增强：`(x_i, augment1(x_i), augment2(x_i))`
- 阳性样本之间的配对（4个阳性样本可组合）

**负样本对：**
- 不同用户之间（特别是阳性-阴性对）

**下游任务：**
- 冻结encoder，只训练一个小分类器
- 对比：encoder表示 vs 原始44维特征

**新建文件：**
- `scripts/10_train_contrastive.py`
- `notebooks/06_contrastive_analysis.ipynb`

**输出：**
- `results/models/contrastive_encoder.pth`
- `results/figures/contrastive_embeddings.png`
- `results/metrics/contrastive_downstream_performance.json`

---

### Phase 5D: Multimodal Transformer/Attention

**目标：** 学习4个模态（GPS、App、通信、活动）之间的交互

#### 5D.1 多模态特征划分

**特征分组（从44维拆分为4个模态）：**
```python
# GPS features: 11维
gps_features = features[:, 0:11]

# App usage features: 10维
app_features = features[:, 11:21]

# Communication features: 11维
comm_features = features[:, 21:32]

# Activity features: 12维 (包含9个activity + 3个phone lock)
activity_features = features[:, 32:44]
```

#### 5D.2 Transformer架构

**新建文件：`src/models/multimodal_transformer.py`**

```python
class MultimodalTransformer(BaseDeepModel):
    """
    将4个模态视为4个token，使用Transformer融合
    """

    def __init__(self, modality_dims=[11, 10, 11, 12], d_model=16, nhead=4):
        # 模态嵌入层（将不同维度投影到统一维度）
        self.modality_embeddings = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in modality_dims
        ])

        # Positional encoding (模态顺序编码)
        self.pos_encoding = nn.Parameter(torch.randn(1, 4, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, 32),  # 拼接4个模态的输出
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x_modalities):
        """
        x_modalities: [batch, 4, varying_dims] 列表
        """
        # 嵌入每个模态
        tokens = []
        for i, (embed, x_mod) in enumerate(zip(self.modality_embeddings, x_modalities)):
            token = embed(x_mod)  # [batch, d_model]
            tokens.append(token)

        # [batch, 4, d_model]
        tokens = torch.stack(tokens, dim=1)
        tokens = tokens + self.pos_encoding

        # Transformer
        attended = self.transformer(tokens)  # [batch, 4, d_model]

        # 拼接并分类
        pooled = attended.flatten(1)  # [batch, 4*d_model]
        logits = self.classifier(pooled)

        # 返回logits和注意力权重（用于可解释性）
        return logits, attended

    def get_attention_weights(self):
        """提取跨模态注意力权重"""
        # 从transformer层提取attention map
        return self.transformer.layers[0].self_attn.attention_weights
```

#### 5D.3 可解释性分析

**注意力权重可视化：**
- 计算每个模态对预测的贡献度
- 示例：GPS模态权重=0.4，App=0.3，通信=0.2，活动=0.1
- 发现：哪个模态对抑郁预测最重要？

**新建文件：**
- `scripts/11_train_transformer.py`
- `notebooks/07_transformer_analysis.ipynb`

**输出：**
- `results/models/multimodal_transformer.pth`
- `results/figures/modality_attention_heatmap.png`
- `results/figures/transformer_feature_importance.png`

---

### Phase 6: 模型对比与评估

**目标：** 系统比较所有模型性能

#### 6.1 统一评估框架

**新建文件：`scripts/12_evaluate_all_models.py`**

```python
# 评估所有7个模型
models = {
    'Logistic Regression': baseline_lr,
    'Random Forest': baseline_rf,
    'XGBoost': baseline_xgb,
    'VAE (Anomaly)': vae_model,
    'GNN': gnn_model,
    'Contrastive': contrastive_model,
    'Transformer': transformer_model
}

# 统一指标
metrics = [
    'AUC-ROC',
    'PR-AUC',  # 更适合不平衡数据
    'Sensitivity (Recall)',
    'Specificity',
    'F1-Score',
    'Permutation Test p-value'
]

# 生成对比表格
results_df = pd.DataFrame(...)
results_df.to_csv('results/tables/model_comparison.csv')
```

#### 6.2 可视化对比

**新建文件：`src/visualization/compare_models.py`**

```python
def plot_all_roc_curves(models, X_test, y_test):
    """绘制所有模型的ROC曲线在同一图上"""
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.legend()
    plt.savefig('results/figures/all_models_roc_comparison.png')
```

**输出：**
- `results/figures/all_models_roc_comparison.png`
- `results/figures/model_performance_barplot.png`
- `results/tables/model_comparison.csv`

---

### Phase 7: 可解释性与生物标记发现

**目标：** 从模型中提取临床可用的洞察

#### 7.1 Baseline模型可解释性

**SHAP分析（已规划在原Phase 5）：**

**新建文件：`src/interpretability/shap_analysis.py`**

```python
import shap

def explain_baseline_models(model, X_train, feature_names):
    """
    为逻辑回归、随机森林、XGBoost生成SHAP值
    """
    explainer = shap.TreeExplainer(model)  # For RF/XGB
    shap_values = explainer.shap_values(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train, feature_names=feature_names,
                      show=False)
    plt.savefig('results/figures/shap_summary.png', bbox_inches='tight')

    # Feature importance ranking
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)

    return feature_importance
```

#### 7.2 深度学习模型可解释性

**新建文件：`src/interpretability/dl_interpretability.py`**

```python
class DeepModelInterpreter:

    @staticmethod
    def vae_latent_space_analysis(vae, X, y):
        """
        分析VAE潜在空间中哪些维度区分阳性/阴性
        """
        with torch.no_grad():
            z_mean, _ = vae.encode(X)

        # 每个潜在维度的判别力（t-test）
        from scipy.stats import ttest_ind
        p_values = []
        for dim in range(z_mean.shape[1]):
            pos = z_mean[y == 1, dim]
            neg = z_mean[y == 0, dim]
            _, p = ttest_ind(pos, neg)
            p_values.append(p)

        return p_values

    @staticmethod
    def gnn_attention_analysis(gnn, graph_data):
        """
        分析GNN注意力权重：哪些用户连接重要
        """
        _, attention_weights = gnn.get_attention_weights()
        # 可视化高注意力边
        return attention_weights

    @staticmethod
    def transformer_modality_importance(transformer, X_modalities):
        """
        计算每个模态的平均注意力权重
        """
        with torch.no_grad():
            _, attended_tokens = transformer(X_modalities)
            # attended_tokens: [batch, 4, d_model]
            modality_norms = torch.norm(attended_tokens, dim=2).mean(0)
            # [4] - 每个模态的重要性分数

        modalities = ['GPS', 'App Usage', 'Communication', 'Activity']
        importance_df = pd.DataFrame({
            'Modality': modalities,
            'Importance': modality_norms.cpu().numpy()
        })
        return importance_df
```

#### 7.3 数字生物标记发现

**综合分析报告：**

**新建文件：`scripts/13_generate_biomarker_report.py`**

```python
# 整合所有模型的特征重要性
biomarkers = {
    'Baseline (SHAP)': top_features_shap,
    'Transformer': top_modalities,
    'VAE': discriminative_latent_dims,
    'GNN': central_nodes_features
}

# 生成临床解释
clinical_interpretation = {
    'location_variance_mean': 'GPS位置方差 ↓ → 社交退缩（Saeb 2015）',
    'call_count_mean': '通话频率 ↓ → 社交孤立（Farhan 2016）',
    'night_usage_ratio': '夜间手机使用 ↑ → 失眠/昼夜节律紊乱',
    'sedentary_days_ratio': '久坐天数 ↑ → 精神运动性迟滞'
}

# 输出markdown报告
report = generate_markdown_report(biomarkers, clinical_interpretation)
with open('results/digital_biomarkers_report.md', 'w') as f:
    f.write(report)
```

**输出：**
- `results/digital_biomarkers_report.md` - 临床可读的生物标记报告
- `results/figures/biomarker_ranking.png` - 跨模型一致性排名
- `results/tables/top_biomarkers.csv`

---

## 文件结构总览

```
multimodal-depression-detection/
├── configs/
│   └── model_configs.yaml              [新建] 超参数配置
├── src/
│   ├── models/
│   │   ├── baseline.py                 [新建] 传统ML模型
│   │   ├── evaluation.py               [新建] 评估工具
│   │   ├── pytorch_base.py             [新建] PyTorch基类
│   │   ├── vae_model.py                [新建] VAE
│   │   ├── gnn_model.py                [新建] GNN
│   │   ├── contrastive_model.py        [新建] 对比学习
│   │   └── multimodal_transformer.py   [新建] Transformer
│   ├── features/
│   │   └── graph_builder.py            [新建] 图构建
│   ├── utils/
│   │   ├── data_loader.py              [新建] 数据加载
│   │   ├── pytorch_utils.py            [新建] PyTorch工具
│   │   └── augmentation.py             [新建] 数据增强
│   ├── interpretability/
│   │   ├── shap_analysis.py            [新建] SHAP
│   │   └── dl_interpretability.py      [新建] 深度学习可解释性
│   └── visualization/
│       └── compare_models.py           [新建] 模型对比可视化
├── scripts/
│   ├── 07_train_baseline.py            [新建]
│   ├── 08_train_vae.py                 [新建]
│   ├── 09_train_gnn.py                 [新建]
│   ├── 10_train_contrastive.py         [新建]
│   ├── 11_train_transformer.py         [新建]
│   ├── 12_evaluate_all_models.py       [新建]
│   └── 13_generate_biomarker_report.py [新建]
├── notebooks/
│   ├── 03_baseline_modeling.ipynb      [新建]
│   ├── 04_vae_analysis.ipynb           [新建]
│   ├── 05_gnn_analysis.ipynb           [新建]
│   ├── 06_contrastive_analysis.ipynb   [新建]
│   └── 07_transformer_analysis.ipynb   [新建]
├── results/
│   ├── models/                         [PyTorch .pth + sklearn .pkl]
│   ├── metrics/                        [JSON性能指标]
│   ├── figures/                        [所有可视化]
│   ├── tables/                         [对比表格]
│   └── digital_biomarkers_report.md    [新建] 最终报告
└── requirements.txt                     [更新] 添加PyTorch依赖
```

---

## 实施顺序建议

### 第一周：Baseline + 基础设施
1. 更新requirements.txt，安装PyTorch（支持MPS）
2. 实现`src/models/baseline.py`和`src/models/evaluation.py`
3. 运行`scripts/07_train_baseline.py`
4. 建立PyTorch基础设施（`pytorch_base.py`, `pytorch_utils.py`）
5. 创建`configs/model_configs.yaml`

### 第二周：VAE + GNN
6. 实现VAE模型（`vae_model.py`）
7. 训练VAE并生成合成样本
8. 构建用户相似性图（`graph_builder.py`）
9. 实现GNN模型（`gnn_model.py`）
10. 训练GNN并可视化节点嵌入

### 第三周：对比学习 + Transformer
11. 实现数据增强策略（`augmentation.py`）
12. 实现对比学习模型（`contrastive_model.py`）
13. 实现多模态Transformer（`multimodal_transformer.py`）
14. 训练两个模型并分析注意力权重

### 第四周：评估 + 可解释性
15. 运行`scripts/12_evaluate_all_models.py`对比所有模型
16. SHAP分析（baseline）
17. 深度学习可解释性分析
18. 生成数字生物标记报告
19. 制作publication-quality figures

---

## 关键技术决策

### 1. 处理小样本（n=46）的策略

**问题：** 深度学习通常需要大量数据，46个样本容易过拟合

**解决方案：**
- ✅ 使用小型网络（参数量 < 1000）
- ✅ 强正则化（Dropout 0.2-0.3, Weight Decay 0.01）
- ✅ Early stopping（patience=20）
- ✅ Leave-One-Out CV而非k-fold（最大化训练数据）
- ✅ 数据增强（VAE合成样本、对比学习augmentation）
- ✅ 迁移学习思想（对比学习预训练 → 微调分类器）
- ✅ 图结构利用用户相似性（GNN半监督学习）

### 2. 处理类别不平衡（4 vs 42）的策略

**问题：** 只有4个阳性样本，模型倾向于预测所有样本为阴性

**解决方案：**
- ✅ Baseline: `class_weight='balanced'`（scikit-learn）
- ✅ XGBoost: `scale_pos_weight=10.5`
- ✅ PyTorch: `WeightedRandomSampler`或focal loss
- ✅ VAE数据增强：生成20-30个合成阳性样本
- ✅ 评估指标：优先看PR-AUC、Sensitivity（而非Accuracy）
- ✅ 阈值调整：优化Sensitivity≥0.80（临床要求）

### 3. MPS加速配置（MacBook）

**Apple Silicon优化：**
```python
# 在所有PyTorch脚本开头添加
import torch

# 自动选择设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 模型和数据移动到MPS
model = model.to(device)
data = data.to(device)
```

**注意事项：**
- MPS在小批量（batch_size < 32）时加速不明显
- 某些操作（如GNN的稀疏矩阵）可能不支持MPS，需降级到CPU
- 建议在`pytorch_utils.py`中封装设备选择逻辑

### 4. 超参数搜索策略

**问题：** 网格搜索成本高，且46个样本不足以分出验证集

**解决方案：**
- ✅ Baseline: 使用文献推荐值（Saeb 2015, Farhan 2016）
- ✅ 深度学习: 先用经验值（configs/model_configs.yaml）
- ✅ 如需调优: 使用嵌套CV（外层LOO-CV，内层5-fold）
- ✅ 优先调整正则化参数（dropout, weight_decay）而非网络结构

---

## 预期结果

### 性能预期（基于文献）

| 模型类型 | 预期AUC-ROC | 优势 | 风险 |
|---------|------------|------|------|
| Logistic Regression | 0.60-0.70 | 可解释性强，baseline | 特征线性假设 |
| Random Forest | 0.65-0.75 | 非线性，鲁棒 | 过拟合风险 |
| XGBoost | 0.65-0.75 | SOTA传统ML | 需仔细调参 |
| VAE | 0.55-0.65 (异常检测) | 生成合成样本 | 重构误差判别力弱 |
| GNN | 0.70-0.80 | 利用用户相似性 | 图结构质量依赖 |
| Contrastive | 0.65-0.75 | 小样本友好 | 增强策略设计难 |
| Transformer | 0.70-0.80 | 跨模态交互 | 参数量大易过拟合 |

**注意：** n=46的小样本会导致AUC置信区间很宽（±0.10），需进行1000次permutation test验证统计显著性（p<0.05）。

### 科研贡献（即使性能一般）

1. **方法学创新：** 首次系统对比传统ML vs 深度学习在小样本多模态抑郁检测任务
2. **表示学习：** VAE/对比学习的潜在空间可视化本身有学术价值
3. **可解释性：** Transformer注意力权重揭示GPS vs 通信 vs 活动的相对重要性
4. **数据增强：** VAE合成样本为未来类似小样本研究提供方法论
5. **临床翻译：** SHAP + 注意力权重 → 可操作的数字生物标记

---

## 风险缓解

### 风险1: 所有模型性能接近随机猜测（AUC~0.50）

**可能原因：**
- 44个特征不包含预测信号
- 4个阳性样本不足以学习模式

**应对措施：**
- 检查特征分布：阳性vs阴性是否有显著差异（t-test）
- 使用permutation test：即使AUC低，也验证是否显著优于随机
- 降级研究问题：从"预测"改为"探索性分析"
- 关注特征表示质量而非分类性能（VAE潜在空间、GNN嵌入）

### 风险2: 深度学习模型严重过拟合

**症状：** 训练AUC=1.0，测试AUC<0.5

**应对措施：**
- 减小模型（hidden_dim从32降到16）
- 增强正则化（dropout从0.2升到0.5）
- 使用更简单的baseline（线性模型）
- 尝试transfer learning（使用大规模健康数据预训练）

### 风险3: MPS加速不工作或报错

**应对措施：**
- 降级到CPU（batch_size小时速度差异不大）
- 检查PyTorch版本（需>=2.0）
- 某些操作手动指定`.to("cpu")`（如稀疏矩阵）

---

## 检查清单

在开始实施前确认：

- [ ] mamba环境`qbio`已激活
- [ ] 确认`data/processed/features/combined_features.parquet`存在（46×44）
- [ ] 确认`data/processed/labels/item9_labels_pre.csv`存在
- [ ] 创建`configs/`文件夹
- [ ] 更新`requirements.txt`添加PyTorch
- [ ] 测试MPS是否可用：`python -c "import torch; print(torch.backends.mps.is_available())"`
- [ ] 阅读Saeb et al. (2015)和Farhan et al. (2016)了解预期特征重要性

---

## 最终交付物

### 代码
- 20+个新Python模块（模型、工具、可视化）
- 7个训练脚本
- 5个Jupyter notebooks

### 结果
- 7个训练好的模型（.pkl + .pth）
- 15+个可视化图表（ROC、confusion matrix、attention、t-SNE等）
- 模型对比表格（CSV）
- 数字生物标记报告（Markdown）

### 文档
- 更新README.md添加Phase 3-7说明
- 每个脚本的docstring和使用示例
- `digital_biomarkers_report.md`包含临床解释

---

## 下一步行动

1. **用户确认计划** - 是否同意上述方案？有调整需求吗？
2. **环境准备** - 安装PyTorch和相关依赖
3. **开始Phase 3** - 先实现baseline模型验证数据质量
4. **迭代开发** - 按周实施VAE → GNN → 对比学习 → Transformer

准备好后我们开始实施！🚀
