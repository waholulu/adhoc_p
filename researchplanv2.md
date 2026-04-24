

# 研究计划书：出院后 HHS 早启动效应与高受益人群识别，v3 推荐版

## 1. 研究目的

本研究评估 Medicaid 出院回家成员中，**出院后前 2 天内启动 Home Health Service，HHS** 是否能降低随后 58 天的再住院和医疗成本，并进一步识别哪些成员最可能从 HHS 早启动中获益。

本研究不把重点放在证明 general population 的平均效果上。平均效果可能不明显。真正的核心是：

> 在可信的因果框架下，找出最适合 CM 团队优先推动 HHS 早启动的高受益成员。

核心问题分为三层：

1. **ATE 背景结果**
   整体人群中，early HHS 相对于 no early HHS 是否降低 58 天再住院和总成本？

2. **HTE 核心结果**
   哪些 subgroup 或 member profiles 的收益明显更高？

3. **业务转化结果**
   CM 团队能否用简单、当天可见的规则，优先识别这些高受益成员？

---

## 2. Target Trial 设定

### 2.1 时间锚点

| 元素        | 定义                             |
| --------- | ------------------------------ |
| T0        | Index inpatient discharge date |
| Landmark  | T0+2 天末                        |
| Follow up | T0+2 天末到 T0+60，共 58 天          |

采用 T0+2 landmark 的原因是：前 2 天内发生再住院或死亡的成员，结构上很难进入 early HHS 组。landmark 设计把两组都限制在 T0+2 天末仍在社区、仍可观察的人群中，使比较更对称。

这个设计的 estimand 是：

> 在 T0+2 天末仍存活、仍参保、未再住院、未转 hospice 或 LTC 的成员中，前 2 天内启动 HHS 相对于未在前 2 天内启动 HHS，对之后 58 天结局的影响。

注意，这不是所有出院成员从 T0 开始的总效果。前 2 天内发生再住院或死亡的人群不在主分析范围内，会作为 cohort flow 和局限性单独汇报。

---

## 3. Cohort 定义

### 3.1 T0 层面纳入

纳入：

1. Medicaid 成员
2. 急性 inpatient discharge
3. Discharge to community setting
   包括 home, self care, home with home health
4. T0 时仍参保
5. 有足够历史观察窗口生成 baseline covariates
6. 分析单位优先使用 member 的第一个 eligible discharge

排除：

1. T0 前死亡
2. AMA discharge
3. 转 hospice
4. 转 SNF, inpatient rehab, LTC 或其他机构
5. 非社区出院
6. 数据质量明显异常的 episode

为了简化主分析，建议首版优先采用：

> 每个 member 只取第一个 eligible discharge。

如果样本量不足，再允许多个 episode，并用 member level cluster bootstrap。

### 3.2 T0+2 landmark 纳入

T0+2 天末必须满足：

1. 仍存活
2. 仍参保
3. T0 到 T0+2 无 inpatient readmission
4. 未转 hospice
5. 未转 LTC
6. 仍处于社区或居家状态

### 3.3 连续参保

主分析建议要求：

> T0 到 T0+60 连续参保，死亡除外。

如果连续参保要求导致样本损失较大，可以做敏感性分析，允许 censoring 并报告影响。

---

## 4. 暴露定义

主暴露不再叫 Early vs Late，而是：

> **Early HHS vs No Early HHS**

| 组别           | 定义                                     |
| ------------ | -------------------------------------- |
| Early HHS    | T0 到 T0+2 天内存在 HHS claim 的 start date  |
| No Early HHS | T0 到 T0+2 天内不存在 HHS claim 的 start date |

No Early HHS 组不是纯粹的 late HHS 组。它包含：

1. T0+3 到 T0+7 启动 HHS
2. T0+8 到 T0+30 启动 HHS
3. T0+31 到 T0+60 启动 HHS
4. 60 天内没有 HHS

这些组成必须做描述性汇报，但主分析只比较：

> 前 2 天启动 vs 前 2 天未启动。

---

## 5. 结局定义

### 5.1 主结局

主结局保留两个：

| 结局                      | 定义                                            |
| ----------------------- | --------------------------------------------- |
| 58 天再住院                 | T0+2 到 T0+60 的非计划 inpatient readmission       |
| 58 天 total allowed cost | T0+2 到 T0+60 的 total allowed cost，包含 HHS cost |

Total cost 包含 HHS cost，因此它本身就是净成本影响，不需要再额外扣除 HHS cost。

### 5.2 成本分解

为了业务解释，再把 total cost 拆成两部分：

| 成本项                     | 目的                                       |
| ----------------------- | ---------------------------------------- |
| Non HHS downstream cost | 看 HHS 是否减少后续 IP, ED, SNF, outpatient 等成本 |
| HHS cost                | 看 early HHS 增加了多少 HHS 本身成本               |

最终汇报时同时展示：

```text
Total net cost impact = change in total allowed cost including HHS
```

以及：

```text
Cost decomposition = non HHS downstream cost change + HHS cost change
```

这样既避免 double counting，也方便老板理解 ROI 来源。

### 5.3 次结局

次结局只保留少数业务有用指标：

1. ED visit
2. SNF admission
3. Days at home
4. 30 天 readmission，T0+2 到 T0+30

---

## 6. Phase 1 Data Readiness Gate

Phase 1 只查最关键的 4 件事，不做过度工程化。

| 检查项             | 标准                                 | 不达标处理                          |
| --------------- | ---------------------------------- | ------------------------------ |
| HHS claim 识别    | 用已知 HHS plan case 反查，命中率最好大于 90%   | 扩充 code list 或重定义 HHS 范围       |
| HHS start date  | `srv_start_dt` 能基本代表 start of care | 如果不可靠，换 first billed date 做敏感性 |
| Readmission 数据源 | Auth 与 claim 一致性最好大于 95%           | 不足则 claim 做主，auth 做敏感性         |
| CM 数据           | 至少能识别 T0 前是否有 CM touch             | 介入类型识别不稳定时，只用 yes/no           |

这里不建议把 CM 介入类型分类作为硬熔断。首版只要能识别：

> T0 前 30 天是否有 CM touch

就可以做主分析。更细的 discharge planning, TOC, disease management 类型可以作为增强版，不作为首版成败关键。

---

## 7. ATE 分析设计

ATE 是背景结果，不是全文重心。

### 7.1 Propensity score

使用 logistic regression 估计 early HHS 的 propensity score。

协变量只使用 T0 或 T0 前可知的信息：

1. 年龄、性别、dual status、产品线、地区
2. 出院月份、年份、周末或节假日前出院
3. Index stay LOS, ICU, DRG 或 MDC, major procedure
4. CHF, COPD, CKD, diabetes, dementia, cancer 等慢病标签
5. 过去 6 到 12 个月 IP, ED, SNF, HHS, DME, total cost
6. T0 前 30 天是否有 CM touch
7. T0 前是否有 PAC referral 或 SNF auth
8. Hospital 或 facility indicator，若样本量允许

不加入 T0 之后的变量，避免控制 mediator 或 collider。

### 7.2 Weighting

主方法使用：

> Overlap weighting

原因是它对应的是最可干预的灰色地带人群，也就是 early HHS 与 no early HHS 都有可能发生的人群。

必须汇报：

1. PS 分布图
2. 加权前后 SMD
3. 加权后的 effective sample size
4. ATO target population 的基本特征

### 7.3 ATE effect estimation

主结果用加权差异，保持简单：

| 结局                      | 估计量                |
| ----------------------- | ------------------ |
| Readmission             | 加权 risk difference |
| Total cost              | 加权 mean difference |
| Non HHS downstream cost | 加权 mean difference |
| HHS cost                | 加权 mean difference |

置信区间建议用 bootstrap。

如果首版每个 member 只取一个 episode，普通 bootstrap 即可。
如果允许多个 episode，使用 member level bootstrap。

成本做 99% winsorization，敏感性用 95% 和 99.5%。

不建议首版主分析使用 Gamma GLM 或 two part model。它们可以作为附录敏感性，不放主线。

---

## 8. HTE 分析设计，核心部分

HTE 是本研究主线。建议分三层做。

---

# HTE 第一层：预定义 subgroup

这是最稳、最快、最容易解释的一层。

预定义 subgroup：

1. CHF
2. COPD
3. CKD 或 ESRD
4. Dementia
5. 过去 12 个月至少 2 次 inpatient admission
6. 过去 12 个月用过 HHS
7. T0 前 30 天有 CM touch
8. 周末或节假日前出院

每个 subgroup 报告：

| Subgroup |  N | Early rate | Δ Readmission | Δ Total Cost | Δ Non HHS Cost | Δ HHS Cost | Priority |
| -------- | -: | ---------: | ------------: | -----------: | -------------: | ---------: | -------- |

分析方式：

1. 使用同一套 PS 和 overlap weights
2. 在 subgroup 内估计加权 outcome difference
3. 检查 subgroup 内 balance
4. 如果 subgroup 样本太小或 balance 很差，标记为 exploratory

这一层的目的不是穷尽所有异质性，而是先回答老板最容易理解的问题：

> 哪些临床或业务上已知的人群看起来更值得优先推动 early HHS？

---

# HTE 第二层：Causal forest ranking

Causal forest 必做，但只承担一个任务：

> 给成员按 predicted benefit 排序，检验 top benefit group 是否真的有更高收益。

不要让 causal forest 同时承担解释、业务规则、policy value、最终部署等所有任务。

### 8.1 主 HTE outcome

HTE 主排序使用：

> 58 天 total allowed cost，包括 HHS cost

原因是它最接近净成本影响，避免 ROI 公式复杂化。

模型直接在 dollar scale 上估计 cost CATE：

```text
tau_cost = E[Cost under Early HHS - Cost under No Early HHS | X]
```

因为 cost 越低越好，所以 predicted benefit 定义为：

```text
Predicted benefit = - tau_cost
```

benefit 越高，表示 early HHS 预计越省钱。

### 8.2 特征集

Causal forest 使用 full feature set，包括：

1. 人口学
2. 慢病标签
3. 历史利用
4. 历史 cost
5. index stay 特征
6. CM touch
7. PAC referral
8. facility 或 hospital 相关特征
9. prior HHS, SNF, DME, oxygen, wound care 等

这些特征用于估计 CATE，不代表业务规则最终都能使用。

### 8.3 验证方式

把数据分为 train 和 holdout，例如 70/30。

在 train 上训练 causal forest。
在 holdout 上按 predicted benefit 分组：

| Predicted benefit group |    人群比例 | 评估内容                         |
| ----------------------- | ------: | ---------------------------- |
| Top 10%                 |    最高受益 | 重点看 total cost 和 readmission |
| Top 20%                 |     高受益 | 业务候选重点人群                     |
| 20% 到 40%               |    中等受益 | 次优先                          |
| Bottom 60%              | 低受益或无受益 | 不优先                          |

每组报告：

1. N
2. Early rate
3. 加权或 doubly robust Δ total cost
4. Δ readmission
5. Δ non HHS cost
6. Δ HHS cost

判断标准：

> 如果 Top 10% 或 Top 20% 的 adjusted total cost saving 明显大于 overall ATE，并且 readmission 没有变差，则说明 HTE 有业务价值。

如果 holdout 上排序不稳定，则 causal forest 结果降级为 exploratory，主 HTE 结论以预定义 subgroup 为主。

---

# HTE 第三层：Operational persona extraction

不要首版强制上 policy tree。最终交付给 CM 团队的应是简单 persona。

做法：

1. 看 causal forest Top 20% benefit group 的成员组成
2. 比较 top group 和 overall population 的特征差异
3. 找出最富集、最容易 operationalize 的特征
4. 转成 2 到 4 个业务 persona

示例：

| Persona                   | 规则                                   | 优先级    |
| ------------------------- | ------------------------------------ | ------ |
| CHF 高利用成员                 | CHF 且过去 12 个月至少 2 次 IP               | High   |
| Prior HHS plus CM touch   | 过去 12 个月用过 HHS 且 T0 前 30 天有 CM touch | High   |
| COPD weekend discharge    | COPD 且周末或节假日前出院                      | Medium |
| CKD plus high utilization | CKD 或 ESRD 且过去 12 个月 ED 或 IP 高利用     | Medium |

每个 persona 报告：

1. 覆盖人数
2. early HHS rate
3. Δ readmission
4. Δ total cost
5. Δ non HHS cost
6. Δ HHS cost
7. 推荐优先级

这一步是最终业务转化重点。

只有当简单 persona 无法解释 top benefit group，或者业务方要求自动规则时，才把 policy tree 作为 Phase 2 增强。

---

## 9. Sensitivity Analysis

首版只保留 5 个关键敏感性，避免项目失控。

| 敏感性                                   | 目的                   |
| ------------------------------------- | -------------------- |
| Landmark 改为 T0+1 和 T0+3               | 看 48 小时窗口是否稳健        |
| IPTW 替代 OW                            | 看 estimand 改变后方向是否一致 |
| Claim readmission 替代 auth readmission | 验证 readmission 数据源   |
| Cost winsorize 95% 和 99.5%            | 验证极端成本影响             |
| Causal forest holdout ranking         | 验证 HTE 排序是否稳定        |

其他方法，例如 Gamma GLM、two part model、Fine Gray、E value、CCW、policy tree bootstrap、RATE，可以作为后续增强，不放首版主线。

---

## 10. 成功标准

这项研究不应该以 overall ATE 是否显著作为唯一成功标准。

建议成功标准设为：

### 10.1 最低成功标准

1. HHS exposure 能可靠识别
2. Early vs No Early 在 OW 后 balance 良好
3. 能给出可信的 overall ATE 背景结果
4. 至少部分预定义 subgroup 显示更强的正向信号

### 10.2 理想成功标准

1. Top 20% predicted benefit group 的 total cost saving 明显高于整体人群
2. Top group 的 readmission 没有上升，最好下降
3. Top group 能被 2 到 4 个 operational persona 解释
4. 每个 persona 覆盖人数足够，最好覆盖 eligible population 的 10% 到 30%
5. CM 团队确认这些 persona 可以放进 work queue 或 outreach priority

### 10.3 降级标准

如果 ATE 不明显，但 subgroup 或 top predicted benefit group 有稳定信号：

> 结论是 early HHS 不适合全量推动，但适合对特定高受益成员优先推动。

如果 causal forest 不稳定，但预定义 subgroup 有信号：

> 结论以 subgroup 为主，ML HTE 作为探索性结果。

如果 subgroup 和 causal forest 都没有稳定信号：

> 结论是现有数据下无法支持 early HHS 优先干预策略，建议回到 HHS identification、SOC timing、CM workflow 或目标人群定义上重新设计。

---

## 11. 项目时间线

建议控制在 6 到 7 周。

| Phase | 内容                                           |    时间 |
| ----- | -------------------------------------------- | ----: |
| 1     | Data readiness gate                          |   1 周 |
| 2     | Cohort, exposure, outcome, covariates        | 1.5 周 |
| 3     | ATE, OW balance, cost decomposition          |   1 周 |
| 4     | Predefined subgroup HTE                      |   1 周 |
| 5     | Causal forest ranking and holdout validation | 1.5 周 |
| 6     | Persona extraction and final report          |   1 周 |

如果数据抽取已经成熟，可以压到 5 周左右。

---

## 12. 最终交付物

建议最终报告控制在 8 到 10 页。

### 主报告结构

1. 研究问题和设计概览
2. Cohort flow，包括 landmark 排除人群
3. Early HHS vs No Early HHS 定义
4. Baseline balance 和 ATO target population
5. Overall ATE 背景结果
6. Cost decomposition
7. Predefined subgroup HTE
8. Causal forest top benefit group 验证
9. Operational persona 推荐
10. 局限性和下一步行动

### 核心图表

1. Cohort flow diagram
2. PS overlap plot
3. Balance love plot
4. ATE result table
5. Subgroup HTE forest plot
6. Causal forest predicted benefit decile table
7. Persona priority table

---

## 13. 我建议你从 v2 删除或降级的内容

| v2 内容                            | v3 处理                  |
| -------------------------------- | ---------------------- |
| Gamma GLM 与 two part 条件分支        | 降级为附录敏感性               |
| Fine Gray competing risk         | 删除首版                   |
| E value                          | 删除首版                   |
| CCW 方法学对照                        | 删除首版或放后续               |
| log scale CATE 对照                | 删除首版                   |
| RATE                             | 可选，不作为核心交付             |
| Policy tree                      | Phase 2 增强，不作为首版承诺     |
| CATE optimal vs ops feasible gap | 删除首版                   |
| CM 介入类型硬熔断                       | 改为增强项，主分析只要求 CM yes/no |

---

## 14. 最终一句话版本

> 本研究用 T0+2 landmark 和 overlap weighting 建立一个可信的因果比较框架，比较前 2 天启动 HHS 与未前 2 天启动 HHS 对随后 58 天再住院和总成本的影响。由于整体平均效果可能不明显，研究重点放在 HTE：先用预定义临床和业务 subgroup 找可解释信号，再用 causal forest 对成员按预期净成本收益排序，并在 holdout 中验证 top benefit group 是否真的更受益。最终把高受益群体转化为 2 到 4 个 CM 团队可执行的 operational persona，用于优先推动 HHS 早启动。

我觉得这版是最平衡的：**足够因果、足够 HTE、足够业务可执行，同时不会被复杂方法拖死。**
