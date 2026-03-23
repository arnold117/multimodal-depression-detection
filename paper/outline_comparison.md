# 论文大纲对比：旧框架 vs 新框架

## 旧框架（正面叙事）

**标题**: Personality Traits Predict Mental Health Across Three Universities: A Multi-Study ML Investigation (N=1,559)

```
1. Introduction
   - Personality → MH literature (已知)
   - Passive sensing 的 promise
   - Gap: 缺少多研究复制

2. Results
   - 人格预测心理健康 ✓
   - Meta-analysis 汇总 ✓
   - 临床分类 AUC ✓
   - 增量效度 ✓
   - SHAP vs 传统方法 ✓
   - Robustness ✓

3. Discussion
   - N 是跨诊断标记
   - Sensing 价值有限
   - 简单模型就够
```

**问题**:
- "N predicts MH" 是 Kotov (2010) 就确认的 → reviewer 说 "not novel"
- 每个分析都有结果，但没有一个统一的 research question 串起来
- Negative findings (sensing 没用, SHAP=β, MLP<Ridge) 散落各处，像是副产品
- 论文试图 claim 太多东西，什么都说一点，什么都不深入

---

## 新框架（对比叙事）

**标题**: Can Passive Smartphone Sensing Replace Personality Questionnaires for Mental Health Prediction? Evidence from Three University Samples (N=1,559)

```
1. Introduction
   1.1 The digital phenotyping promise                    ← 搭建靶子
       - Insel (2017): "smartphones as the next stethoscope"
       - Torous et al. (2021): passive sensing for MH monitoring
       - 隐含假设: 连续的行为数据 > 低频的自报告问卷

   1.2 The personality questionnaire as "dumb baseline"   ← 引入对手
       - Kotov et al. (2010): N → MH meta-analysis
       - 问卷被认为"old school"，5分钟，不够fancy
       - 但从未被系统性地和sensing比较过

   1.3 The missing head-to-head comparison                ← Gap = 我们的入口
       - 多数 sensing 论文不包含人格作为 baseline
       - 多数人格论文不包含 sensing 数据
       - 我们恰好有三个数据集同时包含两者

   1.4 Present study                                      ← 清晰的 RQ
       RQ1: 人格问卷 vs 被动传感，谁更能预测心理健康？
       RQ2: 传感数据能否在问卷基础上提供增量效度？
       RQ3: 更复杂的模型（MLP, ensemble）能否弥补传感的不足？

2. Method (基本不变)

3. Results — 每一节回答一个子问题
   3.1 人格问卷的预测力有多强？（baseline）
       → R²=0.09-0.57, AUC=0.65-0.80, meta r=0.44-0.63
       → 为什么重要: 这是传感需要beat的benchmark

   3.2 被动传感单独能预测吗？
       → R²≤0, AUC=0.53-0.60 (接近随机)
       → 核心对比: 问卷完胜

   3.3 传感能增强问卷吗？
       → FDR后仅1/8显著 (S3 BDI-II ΔR²=0.024)
       → S2例外: Pers+Beh AUC=0.83-0.86 (+0.06-0.08)
       → Nuanced: 数据质量高时有边际贡献

   3.4 更复杂的模型能救吗？
       → MLP<Ridge (regression), MLP<LR (classification)
       → SHAP=OLS β (τ=0.94)
       → 不是模型不够好，是信号本身不在sensing里

   3.5 Robustness
       → Gender控制后人格依然robust
       → COVID敏感性 <0.015
       → BFI-10 vs BFI-44 一致

4. Discussion
   4.1 问卷 vs 传感：一个不对称的竞赛
       - 5分钟问卷 >> 数周连续传感
       - 原因: 人格是stable trait, sensing捕获的是state noise

   4.2 什么时候传感有用？（不完全否定）
       - S2 Pers+Beh AUC 0.85: 高质量Fitbit+通信数据有边际贡献
       - 但这个"高质量"要求很苛刻

   4.3 对 digital phenotyping 领域的启示
       - 需要报告 "questionnaire baseline" 作为标准做法
       - 单独报sensing R²可能夸大其独特贡献

   4.4 简单模型就够了
       - SHAP/MLP/Optuna 都没有发现 Ridge 找不到的东西
       - 方法论的复杂度应该匹配数据的复杂度

   4.5 Limitations
```

---

## 为什么新框架更好？逐点对比

| 维度 | 旧框架 | 新框架 |
|------|--------|--------|
| **核心问题** | "人格预测MH"（已知） | "传感能否替代问卷？"（未回答） |
| **新意来源** | 复制已知发现 | 提出新比较 + 系统回答 |
| **Negative findings** | 尴尬的附带发现 | 论文的核心贡献 |
| **叙事统一性** | 6个分析各自为政 | 所有分析回答同一个问题 |
| **S1的角色** | 撑不起的"Study 1" | Discovery sample（合理化小N） |
| **实际贡献** | "又一个复制研究" | "领域级别的reality check" |
| **目标读者反应** | "嗯，知道了" | "确实没人系统比过" |
| **S2 Pers+Beh 高AUC** | 附带发现 | Nuanced conclusion的关键证据 |

---

## 和老师沟通的建议

老师觉得原先结果 promising，这不矛盾。关键是 **framing 的区别**：

**不要说**："我觉得我们的结果其实是 negative 的"
**要说**："结果本身不变，但我在想怎么写能发更好的期刊"

具体话术：

> 何老师好，
>
> 关于论文框架，我有一个想法想和您讨论。
>
> 我们的核心发现——人格强力预测心理健康——确实很solid。但如果单独以此为卖点，
> reviewer可能会说这和Kotov (2010) 的meta-analysis结论一致，新意有限。
>
> 我在想能不能把角度转一下：既然我们三个数据集**同时**有人格问卷和被动传感数据，
> 这在文献里其实很少见。大多数sensing论文不包含人格baseline，大多数人格论文没有
> sensing数据。我们可以做一个**系统的head-to-head比较**：
>
> - 问卷 AUC 0.65-0.80 vs 传感 AUC 0.53-0.60
> - 传感增量效度 FDR后仅 1/8 显著
> - MLP 也救不了传感
>
> 这样论文的贡献就从"又一个复制"变成了"对digital phenotyping领域的实证检验"。
> 人格的强预测力不是目的，而是作为benchmark来衡量sensing的价值。
> 所有正面结果都保留，只是叙事角度转了。
>
> 目标期刊可以考虑 Computers in Human Behavior（IF~9, Q1）或
> JMIR Mental Health（IF~7, Q1），这两个对这类比较性研究很receptive。
>
> 您觉得这个方向可行吗？

**核心策略**: 让老师觉得这是"同样的好结果，更聪明的包装"，而不是"我在否定我们的工作"。

---

## 文献调研（2025-03-10）

### 1. 有没有人做过 questionnaire vs sensing 的 head-to-head 比较？

**结论：没有找到完全一样的。** 没有一篇论文同时满足：(a) Big Five 问卷 + passive sensing, (b) 分别报告两者 AUC/R², (c) formal incremental validity test, (d) 跨多个数据集。

最接近的：

- **Khwaja et al. (2019)** "Passive mobile sensing and psychological traits for large scale mood prediction," *EAI PervasiveHealth*.
  - 比较了问卷+传感的组合预测 mood（不是临床 MH 结局）
  - 结论是 additive/complementary，没有像我们一样发现 sensing 接近随机
  - 会议论文，不是顶刊

- **Busshart et al. (2026)** "Distinguishing Common Digital Phenotyping and Self-Report Parameters for Monitoring and Predicting Depression: Scoping Review," *JMIR mHealth and uHealth* (IF~5).
  - 这是一个 scoping review，比较了 digital phenotyping vs self-report 参数
  - 是 review 不是实证研究 → 说明这个比较空间正在被关注，但还没人用多数据集实证回答
  - **对我们最有利**：证明 gap 存在，而且刚刚有人指出这个 gap

→ **新框架的新意得到确认**

### 2. 支撑 "sensing 被高估" 的 critical / reality check 论文

- **Muller et al. (2021)** "Depression predictions from GPS-based mobility do not generalize well to large demographically heterogeneous samples," *Scientific Reports* (IF~4.6).
  - GPS→depression: 小样本 AUC=0.82, 大样本 AUC=0.57 (接近随机)
  - **直接支撑我们的发现**: sensing 在现实条件下接近 chance

- **Xu et al. (2022)** "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling," *IMWUT/UbiComp*.
  - 497 unique users, 19 ML algorithms
  - 结论: domain generalization "barely any advantage over majority guessing"
  - **注意**: 我们用的就是 GLOBEM 数据集，他们自己都承认泛化差

- **Das Swain et al. (2022)** "Semantic Gap in Predicting Mental Wellbeing through Passive Sensing," *CHI 2022*.
  - 理论解释了为什么 passive sensing 不行：传感器和自报告之间有"semantic gap"
  - 语言特征（psycho-social signal）优于被动生理特征

- **Currey & Torous (2022)** "Digital phenotyping correlations in larger mental health samples: analysis and replication," *BJPsych Open* (IF~4).
  - mindLAMP app, passive features 和 PHQ-9/GAD-7/PSS/UCLA 弱相关

- **Adler et al. (2022)** "Machine learning for passive mental health symptom prediction: Generalization across different longitudinal mobile sensing studies," *PLOS ONE* (IF~3.7).
  - CrossCheck + StudentLife, 500+ users
  - 单研究模型泛化差，合并训练后仍然 modest

- **2025 JMIR scoping review** — 领域现状：中位样本 N=60, 仅 2% 有外部验证, 76% 单设备, 45% 监测不足7天
  - **我们的 N=1559 + 3 数据集远超领域标准**

### 3. "Personality predicts MH" 的饱和度

- **Kotov et al. (2010)** "Linking 'big' personality traits to anxiety, depressive, and substance use disorders: A meta-analysis," *Psychological Bulletin* (IF~23).
  - 175 studies, 851 effect sizes, N up to 75,229
  - Neuroticism d=1.65, Conscientiousness d=-1.01
  - 引用 4000+
  - **这是旧框架的最大敌人**：reviewer 直接说 "known since 2010"

- ML 预测人格→depression 的论文存在，但多数用 sensing 做特征，不做 head-to-head 对比
- 用 Big Five + ML 跨多大学样本预测 MH 的论文 **比想象中少**——多数 ML 论文专注 sensing

→ **旧框架不算完全饱和，但 novelty 明显弱于新框架**

### 4. 必引的基础文献

| 论文 | 期刊 | IF | 角色 |
|------|------|-----|------|
| Kotov et al. (2010) | *Psych Bulletin* | ~23 | N→MH 的 gold standard meta-analysis |
| Insel (2018) | *World Psychiatry* | ~73 | "Digital phenotyping: a global tool for psychiatry" — 我们的靶子 |
| Wang et al. (2014) | *UbiComp* | conf | StudentLife = 我们的 S1 数据来源 |
| Torous et al. (2021) | *World Psychiatry* | ~73 | Digital psychiatry 综述 — 搭建 sensing promise 的背景 |
| Saeb et al. (2015) | *JMIR* | ~7 | 早期 GPS→PHQ-9 相关 (N=28, 仅相关) |
| Stachl et al. (2020) | *PNAS* | ~12 | Smartphone→Big Five (r_median=0.37) — sensing 能预测人格但很 modest |
| Chikersal et al. (2021) | *ACM TOCHI* | ~4 | Passive sensing→depression (N=138 college students) |

### 5. 目标期刊先例

- **Computers in Human Behavior** (IF~9, Q1): 接受 personality + technology 交叉研究
- **JMIR Mental Health** (IF~7, Q1): Busshart et al. (2026) scoping review 刚发在这里，说明编辑部对这个比较话题有兴趣
- **Journal of Affective Disorders** (IF~6, Q1/Q2 边界): 临床角度，可以强调 AUC 和临床筛查
- **Behavior Research Methods** (IF~5, Q1): 方法论角度（SHAP vs β, MLP vs Ridge）

### 6. 总体评估

| 维度 | 旧框架 | 新框架 |
|------|--------|--------|
| **文献空白** | 小 (Kotov 2010 已确认) | **大** (无人做过 head-to-head) |
| **领域时效性** | 一般 | **高** (2025/2026 scoping review 刚指出 gap) |
| **我们的 N** | 大但不突出 (vs Kotov 的 75k) | **碾压** (vs 领域中位 60) |
| **关键盟友论文** | 太多 (= 饱和) | Muller, Xu, Das Swain, Busshart |
| **reviewer 最可能的批评** | "not novel" | "secondary data" (但有3个独立数据集缓解) |
