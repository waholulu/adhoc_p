# 研究计划书：出院后 HHS 早启动效应与精准干预策略（v2）

**修订说明**：本版本相对 v1 的关键修订集中在 §4 Step 1–3。主要变化包括：(1) Causal forest 的 cost outcome 直接在美元尺度建模，去掉 log1p 变换以避免 Jensen's Inequality 引入的按 X 异质性的 bias；(2) Cost outcome 的主分析模型从预先锁定 two-part 改为基于描述性结果的条件分支；(3) Causal forest 与 Policy Tree 使用不同的 feature 集，前者用全特征求 CATE，后者只用 CM 系统当天可见的 operational features 输出业务规则；(4) 新增 §11 Phase 1 风险节点评估，把 CM 数据库中 T0 前 discharge planning 类介入的可识别性列为硬性熔断标准。

---

## 1. 研究目的

采用 Target Trial Emulation 框架，在 T0+2 天 landmark 设计下，评估出院回家成员在前 2 天内启动 HHS 对随后 58 天再住院和总医疗成本的因果影响，并通过因果机器学习识别最能从早启动中获益的子人群，输出可执行的 care management 干预规则。

两个核心业务问题：

1. **ATE**：在出院回家且撑过前 2 天的人群中，平均上早启动 HHS 能否显著降低再住院和成本？
2. **HTE + Policy**：CM 团队应该优先把"出院前或刚出院 48 小时内推动 HHS 早启动"的资源投给哪类成员，ROI 最高？

---

## 2. Target Trial 设定

### 2.1 时间锚点

- **T0**：出院日期（index discharge）
- **Landmark (T_L)**：T0 + 2 天末
- **随访期**：T_L 起算 58 天，总观察窗口 T0 到 T0 + 60 天

**设计逻辑**：由于前 2 天内发生再住院的成员结构性不可能进 Early 组（不在家即无法启动 HHS），两组在 T0 起点会不对称。将 eligibility 的确认锚定到 T_L（两组对称 landmark），确保 Early 和 Late 组在随访起点对齐，避免 selection-into-exposure-by-outcome 偏差。

### 2.2 纳入标准

**T0 层面 eligibility**：
- 研究期内的急性住院出院（auth 数据）
- Discharge disposition = home 或 self-care（T0 字段）
- 出院时连续参保
- 分析单位为 index discharge，设 30 天 washout 去重
- 排除 T0 前死亡、AMA 出院、转 hospice、转 SNF/rehab/LTC、转长期机构

**T_L 层面 eligibility（landmark 条件）**：
- T0+2 天末仍存活
- T0+2 天末仍在参保中
- T0 到 T0+2 天内未发生 inpatient readmission
- T0 到 T0+2 天内未转入 hospice 或 LTC

**描述性汇报（不影响分析但必须交付）**：在 T0 eligible 人群中，因 landmark 条件被排除的 episode 规模（预计 1–3%）、基线特征、排除原因分布。让审阅者看到被排除人群的规模和性质。

### 2.3 暴露定义

**数据源**：HHS claim，`srv_start_dt` 作为启动日期锚点。

- **Early 组**：T0 到 T0+2 天内存在 HHS claim 的 `srv_start_dt`（且通过 landmark）
- **Late 组**：T0 到 T0+2 天内不存在 HHS claim 的 `srv_start_dt`（且通过 landmark）

**HHS claim 识别**：通过 HHA place of service、revenue code（0550–0559 等）、provider type 组合识别，Phase 1 确认准确 code 清单并抽样核对。**注意 Medicaid 人群里要覆盖 Medicare certified HHA、state Medicaid personal care、managed LTSS 下的 HHS、waiver program 下的 HHS 等多种来源**（详见 §11）。

**两组对称性保证**：由于 landmark 排除了前 2 天 readmission/死亡，Early 和 Late 两组都由"T0+2 天末仍在家"的人构成，不存在"Late 组混入必然住院者"的结构性不对称。

**Claim runoff 保护**：研究期截止后等 6 个月以上再拉数据，做 HHA claim completeness check，确认最近期 HHS volume 已稳定。

### 2.4 随访与结局

- **随访窗口**：T_L（T0+2 天末）到 T0+60 天，共 58 天
- **主结局 1 — 再住院**：58 天内非计划 inpatient readmission
  - 数据源：Auth（主），facility claim（验证）
  - Planned readmission：若 auth 无 planned flag，用 CMS planned readmission algorithm 基于 facility claim 剔除
  - Phase 1 做一次 auth-claim 再住院一致性 reconciliation，目标 >95%
- **主结局 2 — 成本**：T_L 到 T0+60 天期间的 total allowed cost（medical + pharmacy claim），99 百分位 winsorize
- **次结局**：30 天（T_L 到 T0+30）readmission、ED visit、SNF admission、post-acute cost、days at home
- **Competing risk**：死亡。主分析用 readmission-or-death 复合终点；敏感性用 Fine-Gray 看 cause-specific readmission

### 2.5 Data Readiness Gate（Phase 1 第一周）

**任一不达标需回炉方案**：

1. **CM 介入类型可识别性（硬性熔断，详见 §11）**：T0 前 30 天 CM 介入是否能识别为 discharge planning / transitions of care / disease management 三类，自动分类与人工标注一致率 ≥80%
2. **HHS claim 识别清单**：和数据/业务 owner 确认准确的 place of service、revenue code、provider type 组合，抽 30 个 episode 手动核对；用已知 CM "discharge with HHS plan" case 反查 claim 的命中率应 >90%
3. **`srv_start_dt` 精度**：抽样核对 HHS claim 的 `srv_start_dt` 与实际 SOC 一致性
4. **Auth-based vs claim-based readmission reconciliation**：计算一致性比例
5. **CM 数据库可拉取性**：确认能在日级精度拉取 T0 前后的 CM 介入记录
6. **Landmark 损失评估**：前 2 天再住院/死亡/失保的 episode 占比，评估是否影响样本量
7. **Claim runoff completeness check**

---

## 3. 数据源分工

| 元素 | 数据源 | 备注 |
|---|---|---|
| **HHS 启动 (exposure)** | Claim | `srv_start_dt`；多来源 code 清单见 §11 |
| **再住院 (outcome 1)** | Auth（主）+ facility claim（验证） | 未来可能切换 |
| **成本 (outcome 2)** | Medical + pharmacy claim | — |
| **Discharge disposition** | Facility claim / auth | T0 eligibility |
| **Case management** | 专门 CM 数据库 | 日级时间戳 |
| **Index admission 严重度** | Facility + professional claim，auth 补充 | — |
| **历史利用 (6–12 个月)** | All claim | — |
| **Comorbidity** | Claim → Elixhauser / CMS-HCC | — |
| **Post-discharge referral 意图** | Auth | 非 HHS 的 PAC auth（SNF referral 等） |

---

## 4. 核心分析流水线

### Step 1：ATE 估计

#### 4.1.1 Propensity Score 建模

所有协变量均在 T0 或之前可知，且只包含通过 landmark 的 episode。

协变量清单：

*人口学*：年龄、性别、dual、地区、plan/product、出院月份和年份、周末或节假日前出院

*Index stay*：DRG/MDC、LOS、ICU、ventilator、discharge disposition、major procedures、CMI proxy

*Comorbidity*：Elixhauser 或 CMS-HCC score，关键慢病（CHF、COPD、CKD、ESRD、diabetes、dementia、cancer）

*历史利用 (6–12 个月)*：total cost、IP admits、ED visits、SNF use、prior HHS use、DME、oxygen、wound care、PT/OT、pharmacy burden

*Care management（专门数据库，强混杂控制）*：
- T0 **之前**是否已有 CM 介入（是/否）
- T0 **之前 30 天内**的 CM 活动强度
- CM 介入类型：discharge planning、transitions of care、disease management、其他
- **注**：CM 介入发生在 T0 之后的不进 PS（变成 mediator），仅作独立描述

*Post-discharge referral 意图（auth）*：T0 当天或之前是否有 HHS 以外的 PAC auth（SNF referral、outpatient rehab）

*T0 到 T_L 期间的事件（可选，仅做敏感性）*：主分析不加入，避免 collider bias；敏感性可加入"T0 到 T_L 期间 ED visit"

*供给端*：hospital fixed effect、HHA density、rurality（如可得）

主模型：logistic regression。敏感性：GBM 估 PS。

#### 4.1.2 Weighting：Overlap Weights (ATO)

- 主分析用 OW，对应 CM 能干预的"临床决策灰色地带"成员
- 加权后关键协变量 SMD <0.05
- 交付时明确描述 ATO target population 特征，避免被外推
- 敏感性用 stabilized IPTW（1–99 百分位截断）

#### 4.1.3 Effect Estimation

**Readmission**：加权 logistic regression，报 58 天 RD 和 RR

**Cost（条件分支决策）**：

预先锁定 two-part model 是过度反应。Cost outcome 的主模型选择挂钩到 Phase 2 描述性结果，决策规则如下：

| 58 天成本 = 0 的比例 | 主分析模型 | 敏感性 |
|---|---|---|
| < 15% | 加权 Gamma GLM with log link（0 值加 \$1 常数），HC-robust SE + cluster | Two-part |
| 15–30% | 加权 Gamma GLM with log link，cluster-robust SE | Two-part 作敏感性 |
| > 30% | Two-part model（logistic + Gamma GLM）作主分析 | 单一 GLM 作对照 |

理由：claim cost 包含药费、门诊、professional 所有来源，以 Medicaid RAP 这种慢病基线高的人群，估计 <5% 的 episode 会 58 天零成本。单一 Gamma GLM 在低零比例时标准误推导更直接、bootstrap 更省事。Two-part 在零比例高时才必要。

**标准误**：cluster-robust SE，按 discharging facility 聚类

**备选**：cluster-level bootstrap 1000 次

### Step 2：HTE 估计（CATE）

**Causal Forest（R 包 `grf`）—— 修订：cost outcome 直接在美元尺度建模**

**为何不用 log 变换**：在 log1p 尺度上估的 CATE 是 `E[log(Y_1) − log(Y_0) | X]`，反变换 `expm1()` 回来的不是 `E[Y_1 − Y_0 | X]`（Jensen's Inequality）。这个 bias 在右偏 cost 数据上不仅显著，而且**随 X 异质性变化**——会扭曲 HTE 排序本身，比单纯的全局缩放更麻烦。Duan's smearing 假设残差同方差，claim cost 不满足。

**采用方案**：直接在 winsorized 美元尺度跑 causal forest。森林本身非参数，对分布无要求；`grf` 的 honest splitting + local linear correction 在连续右偏 outcome 上有良好的实证表现；CATE 单位直接是美元，policy tree 的 reward matrix 不需要任何反变换，消灭一整类 bug。

```r
cf_readm <- causal_forest(
  X       = X_full,
  Y       = Y_readm,
  W       = treatment,
  W.hat   = ps_from_step1,
  num.trees = 2000,
  honesty = TRUE
)

cf_cost <- causal_forest(
  X       = X_full,
  Y       = Y_cost_winsorized,   # 直接美元，已 99% winsorize
  W       = treatment,
  W.hat   = ps_from_step1,
  num.trees = 4000,              # cost outcome 噪音大，多种树降方差
  honesty = TRUE
)
```

**敏感性**：保留一版 log 尺度 + smearing 的 CATE 作为方法学对照，但**主分析以美元尺度为准**。

**异质性显著性**：
- **RATE**（`grf` 内置）：按模型排序挑人 vs 随机挑人
- **效应分布图**：tau_hat 直方图
- **Variable importance**：识别 HTE 驱动因素

**过拟合防护**：
- 70/30 train/holdout
- Holdout 上验证 Top-K 高受益人群是否仍高受益
- Holdout RATE 大幅衰减则降级为探索性结论

### Step 3：Policy Rule 输出

**关键设计：Causal Forest 与 Policy Tree 使用不同的 feature 集**

CATE estimation 的目标是**尽可能准确地估个体效应**——用全特征。Policy rule 的目标是**让 CM 团队当天能用**——只能用 work queue 里可见、不需要复杂计算的特征。这两个目标的 feature requirements 是不同的，应分两阶段处理。

#### 4.3.1 Stage A：Full-feature CATE estimation

用 §4.2 的全特征 causal forest（X_full：包括 Elixhauser 组件、CMS-HCC score、详细历史利用、所有 PS 协变量）估 CATE。

#### 4.3.2 Stage B：Operational policy tree

只用 CM 系统肉眼可见、work queue 里可查、不需要实时计算的特征训练 policy tree。

**Policy tree 输入特征清单（X_ops）**：

| 类别 | 特征 |
|---|---|
| 人口学 | 年龄段（<65、65–79、≥80）、性别、dual status |
| 慢病二元标签 | CHF、COPD、CKD、diabetes、dementia |
| 历史利用粗粒度 | 过去 12 个月是否有 ≥2 次 inpatient admission（高利用标签）、过去 12 个月是否用过 HHS |
| Index admission | 是否 ICU、major diagnostic category（不用具体 DRG） |
| CM 上下文 | T0 前 30 天内是否有 CM 介入、CM 介入类型（discharge planning 等） |
| 时点 | 出院是否周末 |

**不进 policy tree 的特征**：
- Elixhauser score 连续值（组件的二元标签可以）
- CMS-HCC risk score
- 具体 DRG 代码
- 历史 cost 连续值（但分位数段或"高利用"二元标签可以）
- 任何需要实时计算的衍生特征

**代码实现**：

```r
# Stage A: full-feature CATE
cf <- causal_forest(X_full, Y, W, W.hat = ps)

# Stage B: policy tree on operationally-accessible features only
X_ops <- covariates[, operational_feature_list]
gamma <- double_robust_scores(cf)
tree  <- policy_tree(X_ops, gamma, depth = 2)
```

输出最多 4 个叶子的决策树，每叶对应一类成员 persona，标注"CM 优先推动早 HHS"或"不优先"。

**树的稳健性**：Bootstrap 100 次重训，top-level split 变量需在 >80% bootstrap 里一致。

**理论依据**：Athey & Wager (2021) 的 policy learning literature 明确背书——policy 学习的输入应该是 "available at decision time" 的 feature，不是所有能预测 CATE 的 feature。

#### 4.3.3 Policy Value 评估（含 HHS 增量成本，并对比 CATE-optimal 上限）

| 策略 | 覆盖人群 | Δ Readmission | Δ Downstream Cost | HHS 增量成本 | **Net Savings** |
|---|---|---|---|---|---|
| 现状 | — | baseline | baseline | baseline | baseline |
| 全量推动早启动 | 所有 landmark 通过者 | −X% | −\$A | +\$B | −\$A + \$B |
| **CATE-optimal**（理论上限） | 全特征排序 top-K | −Y₁% | −\$C₁ | +\$D₁ | **−\$C₁ + \$D₁** |
| **Ops-feasible**（policy tree） | Persona 子群 | −Y₂% | −\$C₂ | +\$D₂ | **−\$C₂ + \$D₂** |

**CATE-optimal vs Ops-feasible 的 gap = "简化成本" (cost of interpretability)**。让老板看到为可执行性放弃了多少精度。这个 gap 通常在 10–20% 之间，完全值得。如果 gap >35%，需要重新检视 operational feature 集合是否漏掉了关键 HTE 驱动因素。

**HHS 增量成本**：从 claim 直接算两组 58 天内的 HHS 总 allowed cost 之差。Net savings 是真实 ROI。

---

## 5. 敏感性分析

1. **Landmark 时点**：T0+1 天末、T0+3 天末、T0+5 天末，看结论是否稳健
2. **Exposure 启动日期**：`srv_stop_dt` 或 first billed date 替代 `srv_start_dt`
3. **Outcome 定义**：30 天 readmission（T_L 到 T0+30）、90 天 readmission、不同 planned readmission 剔除口径、cost winsorize 阈值（95%、99.5%）
4. **Cost outcome 模型**：若主分析用单一 Gamma GLM，跑 two-part 作敏感性；反之亦然
5. **Cost CATE 尺度**：log1p + smearing 反变换 vs 美元尺度直接建模，对比 Top-K 排序的 rank correlation
6. **Readmission 数据源**：auth-based vs claim-based（为未来切换做准备）
7. **Weighting**：IPTW 替代 OW，比较 estimand
8. **PS 模型**：GBM 替代 logistic
9. **Matching 验证**：PSM 1:1 做方向性验证
10. **Competing risk**：Fine-Gray for readmission
11. **CCW 方法学对照**：CCW 版本作为方法学验证——若主 landmark 分析和 CCW 结论方向一致、量级相近，landmark 结论稳健
12. **预定义亚组**：CHF、COPD、高利用、周末出院、有/无 T0 前 CM 介入、产品线
13. **未测混杂**：E-value

---

## 6. 实施步骤

| Phase | 内容 | 时长 |
|---|---|---|
| 1a | **Data Readiness Gate（§2.5 + §11 风险评估）** | 1 周 |
| 1b | 数据抽取、协变量生成、claim runoff check | 2 周 |
| 2 | 描述性分析、cohort flow、landmark 损失描述、crude outcomes、**cost = 0 比例评估（决定 cost 模型分支）** | 1 周 |
| 3 | Step 1：PS + OW + cost model（按分支）+ cluster SE | 1.5 周 |
| 4 | Step 2 + 3：Causal forest（美元尺度）+ ops-feature policy tree + policy value（含 CATE-optimal 上限对比） | 2 周 |
| 5 | 敏感性分析（含 CCW 对照、log-scale CATE 对照） | 1.5 周 |
| 6 | 汇报准备 | 1 周 |

总时长约 10 周。

---

## 7. 交付物

**核心报告（9 页）**：

1. 研究问题 + 设计总览（含 landmark 设计理由）
2. Data readiness 结果 + cohort flow（T0 eligible → T_L eligible → 分组）
3. **Landmark 损失描述**：被排除的前 2 天 readmission/死亡 episode 的规模和特征
4. Baseline balance（加权前后 SMD）+ ATO target population 描述
5. **ATE 主结果**：58 天 readmission RD + cost Δ（cluster SE CI）；注明 cost 模型分支选择依据
6. **HTE 分析**：效应分布 + RATE + holdout 验证（说明 cost CATE 在美元尺度建模的理由）
7. **Policy Tree 图**：2–4 个业务可读 persona（明示只用 operational features）
8. **Policy Value 表**：含 HHS 增量成本的 net savings + CATE-optimal vs Ops-feasible 对比 + 简化成本量化
9. 敏感性分析汇总（含 CCW 方法学对照、log-scale CATE 对照、cost 模型对照）+ 局限性

**附录**：完整敏感性结果、预定义亚组、代码规格、data reconciliation、landmark 时点稳健性、operational feature 清单详解

---

## 8. 关键方法学决策

| 决策 | 理由 |
|---|---|
| **Landmark 到 T0+2 天末，两组对称 eligibility** | 前 2 天 readmission 结构性不可能进 Early，landmark 消除两组不对称性 |
| Exposure 用 claim `srv_start_dt` | 业务现状，未来可补 auth |
| Readmission 用 auth 主、claim 验证 | 时效性 + 有 fallback |
| Cost 用 claim | 唯一可信来源 |
| CM 作为协变量（T0 前） | 独立数据库，避免 mediator |
| **Overlap Weights 主分析** | 对应 CM 能干预的灰色地带人群 |
| **Cost 模型按零比例条件分支** | 避免预先锁定 two-part 的过度反应；零比例低时单一 Gamma GLM 更省事且 SE 推导更直接 |
| **Cluster-robust SE 按 facility** | 同院 HHS 模式相关 |
| **Causal forest 在美元尺度建模 cost CATE** | 避免 Jensen's Inequality 引入的随 X 异质性 bias 扭曲 HTE 排序；CATE 单位即美元，policy tree reward 不需反变换 |
| **Policy tree 只用 operational features** | CATE estimation 与 policy learning 的 feature 需求不同；policy 输入必须是 decision time 可见的特征 |
| **Policy value 同时报 CATE-optimal 与 Ops-feasible** | 量化"简化成本"，让老板看到可执行性的代价 |
| Causal forest + Policy Tree | 从 ATE 升级到"谁最该被推" |
| Policy Value 含 HHS 增量成本 | 给业务真实 ROI |
| CCW 作为敏感性而非主分析 | Landmark 实现简单且审计友好，CCW 验证结论稳健性 |

---

## 9. 业务对接：从结论到动作

研究结论服务于一个明确的 CM 动作：**在出院前或刚出院 48 小时内，CM 团队对高受益成员优先推动 HHS 早启动**。推动方式包括：
- 出院前联系 discharge planner 提前下 HHS order
- 对接 preferred HHA，请求加急接单
- 出院当天成员 outreach，促使家庭尽快接受 HHS

Policy Tree 输出的 persona 直接对接 CM work queue 优先级。由于 policy tree 只用 operational features，CM 团队可以直接把规则写进 work queue 的优先级排序逻辑，无需依赖任何模型实时打分。

**与现有 risk stratification 的关系**：若 CM 团队已有 30 天 readmission 风险分层，policy tree 输出的"早 HHS 高受益"规则可以作为 overlay（谁最该被干预的 second dimension），不是替代。最终交付前和 CM 运营团队对齐集成方式。

---

## 10. 局限性

1. **Landmark 的 estimand 边界**：结论适用于"出院回家且撑过前 2 天"的人群。前 2 天 readmission 的风险因素（被排除）无法通过本研究回答。若前 2 天 readmission 率较高（>5%），结论的泛化性需降级
2. **ATO target population 的外推边界**：结论适用于决策灰色地带人群，不能外推到必然早启动或必然不启动的成员
3. **未测混杂**：功能状态（ADL）、家庭照护资源、patient preference 在 claim 和 CM 数据里都缺失，E-value 量化
4. **HHS 异质性**：服务强度、类型（skilled nursing vs PT/OT vs HHA aide）未细分
5. **HHS 无 auth 数据**：启动决策链路上谁是真正影响因素（discharge planner / HHA / 成员）无法区分
6. **Claim runoff**：最近期研究数据可能小幅上修，buffer + completeness check 缓解
7. **Readmission 数据源未来切换**：auth → claim 切换时需重跑验证
8. **推广性**：Medicaid RAP 人群，其他产品线需单独验证
9. **Operational policy 的精度损失**：Policy tree 只用 ops features，相对 CATE-optimal 有 10–20% 的 policy value loss，是为可执行性付的代价
10. **Cost CATE 在美元尺度建模的代价**：相对 log 变换，对极端高成本 episode 的相对效应不敏感；用 winsorize 缓解，敏感性里跑 log + smearing 对照

---

## 11. Phase 1 风险节点评估

按风险从高到低排列，Phase 1 第一周需要逐项核查。

### 11.1 高风险（最可能熔断）：CM 数据库在 T0 时点的时间戳精度和介入类型分类

**问题**：CM 系统通常是半结构化的，日期字段往往是 "case opened"、"first touch"、"last activity"，不一定有干净的"介入类型" taxonomy。Discharge planning vs transitions of care vs disease management 在原始数据里可能是同一个 case type，需要靠 case note 文本、case reason code、或 workflow stage 推断。

**后果**：如果 T0 前的 CM 介入分不清类型，只能用"是否有任何 CM 介入"这个粗粒度二元变量作为 PS 协变量。对于 CM 介入和 HHS 早启动强相关（discharge planning type）这个场景，粗粒度变量会漏掉关键混杂，ATE 估计可能残留未测混杂。同时 policy tree 也会失去一个关键的 operational feature。

**为何列为硬性熔断**：这一个变量同时是
- PS 模型里最强的 confounder（强关联 exposure 和 outcome）
- 业务上最可能杂乱（半结构化 + 跨 case type）
- 也是未来业务动作的关键依赖（policy tree 如果没有这个 feature，干预点就落空）

**Phase 1 核查动作**：
1. 和 CM 运营团队开一次会，拿 case type 和 reason code 的业务 taxonomy
2. 拉一个小样本（50 个 case）手动打标，验证基于 reason code/workflow stage 的自动分类准确率
3. **熔断标准：自动分类与人工标注一致率 ≥80%**；不达标需重新设计 CM 介入特征（可能要走 case note NLP 或回到粗粒度二元变量并加强其他混杂控制）

### 11.2 中高风险：HHS claim 的 identification code 清单完整性

**问题**：HHA 的 claim 来源多——Medicare certified HHA、state Medicaid personal care、managed LTSS 下的 HHS、waiver program 下的 HHS。不同来源走不同 code 组合。如果只 cover 了主流的 revenue code 0550–0559，可能漏掉 waiver 里的 personal care attendant 类服务，这在 Medicaid 人群里比例不低。

**后果**：Early 组被低估，一部分实际早启动 HHS 的人被错分到 Late 组，属于 non-differential exposure misclassification，**会把 ATE 往 null 方向偏**。

**Phase 1 核查动作**：
1. 用已知的 CM "discharge with HHS plan" case 反查 claim，看能否在 2 天窗口内找到对应 HHS claim
2. **熔断标准**：命中率 >90% 合格；80–90% 需扩充 code 后重测；<80% 需要业务方深度参与重定义 HHS 范围

### 11.3 中风险：Auth-based 与 claim-based readmission 一致性

**问题**：未来可能切换数据源说明系统未完全稳定。

**Phase 1 核查动作**：抽样计算两种来源 readmission 的一致性比例
- ≥95%：主分析用 auth，敏感性跑 claim
- 90–95%：主分析用 auth，敏感性必须报告，结论需注明数据源依赖
- <90%：暂时用 claim 做主分析，等 auth 稳定后再切，或两种都报

### 11.4 中风险：Landmark 损失的实际规模

**问题**：前 2 天 readmission + 死亡 + 失保三者合起来的 episode 占比。

**Phase 1 核查动作**：直接统计
- ≤5%：landmark 设计 estimand 边界可接受
- 5–8%：在局限性单独说明，强调泛化性边界
- \>8%：landmark estimand 边界尴尬，考虑改用 CCW 作主分析，landmark 转敏感性

### 11.5 低风险：HHA claim runoff stability、`srv_start_dt` 精度

工程性问题，runoff buffer 给足（≥6 个月）+ 抽样核对即可。

### 11.6 风险节点优先级总结

| 节点 | 风险等级 | 熔断标准 | 不达标的降级路径 |
|---|---|---|---|
| CM 介入类型分类 | **高（硬性熔断）** | 一致率 ≥80% | 改用粗粒度二元 + 强化其他混杂 + 局限性单列 |
| HHS code 清单 | 中高 | 反查命中率 >90% | 扩充 code 或重定义 HHS 范围 |
| Readmission 一致性 | 中 | ≥95% | 切换主数据源或并报 |
| Landmark 损失 | 中 | ≤5% | 转 CCW 作主分析 |
| Runoff、srv_start_dt | 低 | buffer ≥6 个月 | 延长 buffer |

---

## 一句话总结

**在 T0+2 天末做对称 landmark，两组都限于"出院回家且撑过前 2 天"的成员**，消除 Early/Late 结构性不对称；用 OW + 条件分支的 cost 模型估 ATE；用美元尺度的 causal forest（避免 Jensen's Inequality）+ 仅用 operational features 的 policy tree 估 HTE 并输出可执行规则；最终交付含 HHS 增量成本的 net savings 表，并对比 CATE-optimal 上限以量化可执行性的代价。CCW 作为方法学敏感性验证，如方向一致即证明 landmark 设计稳健。Phase 1 的硬性熔断在 CM 介入类型可识别性。
