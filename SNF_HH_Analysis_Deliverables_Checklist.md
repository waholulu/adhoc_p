# SNF → Home Health 研究：分析与结果交付清单

日期：2026-09-21。用途：研究方案、结果目录、Claude / Codex 执行规范。本文是设计建议，没有分析患者数据，也没有生成实际效果估计。

## 1. 要讲清楚的研究链条

明确出院场景与比较人群 → 证明数据和两组可比性 → 估计重叠人群平均效果 → 检验预设亚组 → 独立验证 causal forest 的排序和筛选能力 → 描述可行动人群及净节约 → 明确不确定性与适用边界。

固定 A=1 为 HH，A=0 为 SNF。ΔR=Risk(HH)−Risk(SNF)，正数表示 HH 再入院更高；ΔC=Cost(HH)−Cost(SNF)，负数表示 HH 节约；Savings=−ΔC。readmission 风险差以百分点报告。HTE 是异质性治疗效果；CATE 是给定特征的条件平均治疗效果，不是已知的个人因果效果。

### 1.1 必须先固定的五个设计决定

1. **主分析直接 PS → overlap weighting（OW）**。PS matching（PSM）放在平行敏感性分析，不默认“先匹配再 OW”。匹配会先改变样本与目标人群；若确实串联，必须重新定义 estimand。不同 estimand 的结果不要求相同。
2. **OW 对应 ATO（重叠人群平均效果），不是所有急性住院出院患者的 ATE。** 两组加权后代表同一个更有可比性的目标人群。PS=e(X)=P(HH|X)，HH 权重=1−e，SNF 权重=e。基础 OW 无需为控制大权重而常规截尾，但仍必须检查局部支持、模型设定和 ESS。
3. **所有实际去 HH/SNF 的比较，不自动等于原计划 SNF 人群的替代效果。** 有 SNF 推荐、转诊、授权申请及其时间戳时，另建明确的“SNF 被考虑”子队列；无这些字段时，研究结论只能是实际照护路径比较及候选机会识别。
4. **两天内启动是出院后的信息。** 不能直接筛选两天内启动者、排除早期失败者后，再声称从出院时起估计了完整 30 天替代效果。见时间设计。
5. **发现与验证必须隔离。** 预设亚组先锁定；模型开发、阈值选择、画像规则发现与最终测试分开。相同患者不能跨训练/测试。honest forest 或 OOB 本身不自动替代所有最终策略的独立验证。

### 1.2 时间设计：先选定一个主版本

| 版本 | 入组和治疗定义 | 随访 | 回答的问题与限制 |
|---|---|---|---|
| A：出院决策版本，有可信的预先计划字段时优先考虑 | t0 为最终急性住院出院；按当时已记录的 HH/SNF 照护计划分组 | 从 t0 计 30/90 天，保留早期事件 | 接近“选择哪种出院计划”的问题；观察性 planned-strategy 比较不是真正随机 ITT；实际落实和 crossover 单独报告 |
| B：48 小时 landmark，贴近当前 claims 方案的可行第一版 | 从原始出院人群开始，在固定 landmark 筛选存活、未发生研究定义的早期再入院，且窗口内已启动 HH/SNF 者；双重服务按预定规则 | 主健康结局建议明确写成 landmark 至出院后第30天；费用 landmark 至第90天，0至landmark费用另报；也可用 landmark 后完整30/90天，但必须换名且全程统一 | 仅是两天内已落实服务且存活未再入院者的条件比较，不能消除所有因 post-treatment selection 导致的偏差，不能覆盖所有出院决策 |
| C：从出院起的 grace-period 策略 | t0 纳入合格者，比较“窗口内开始 HH”与“窗口内开始 SNF”；明确早期事件、启动顺序、偏离策略；可采用 clone-censor-weight | 从出院起完整随访，早期事件不能简单删掉 | 更贴近完整出院替代问题；需纵向数据、人工删失规则和删失权重模型，不能把基础 OW 当作其替代 |

有精确时间戳才称 48 小时；只有日期时用“出院日 D0 至 D2（含）”等明确日历定义，landmark 置于窗口结束后，不能把 D0–D2 自动叫精确 48 小时。使用实际 service date，不使用 claim submission/paid date。HH 以符合研究定义的首次实际专业访视为准，不是 referral/authorization 日期；SNF 以入住/服务开始为准。

### 1.3 候选初版配置，必须参数化

- 研究期：按数据完整期确定，末端预留最长随访与 claims runout；时间范围不能由 AI 编造。
- 人群：成人急性住院最终出院；保险产品分别报告，必要时分层，不混合解释为同一支付制度。
- 基线：入院前365天可观察医疗记录为建议起点；与出院前 index admission 信息分开。药品可用性单独标记。180天基线为敏感性版本。
- 重复住院：第一版每人首个合格 index discharge，另做所有合格 episodes 的版本；后者处理随访窗口重叠、费用重复归属和患者内相关。
- 决策前排除：急性院际转院合并为连续住院 episode；院内死亡；按 scope 排除长期机构居民、基线 hospice、与 HH/SNF 比较不符的专科住院。定义明确的临床排除项而非凭 ICD 名称随意删人。
- 出院后 hospice、再次住院、后续转 SNF/HH 都是结局/路径信息，不能为了让效果好看而删掉。非 landmark 设计不能删除两天内死亡或再入院。
- 随访可观察性：不要求患者必须存活并连续参保满90天。优先保证行政数据截止覆盖随访；失保单独处理删失/观察概率，死亡单独编码。不得把失保后的缺失费用当零。
- 双重启动：窗口内 SNF 与 HH 都出现，先排查 billing overlap/日期错误；主分析可定义互斥单一路径并把混合路径单列。另做首次启动规则敏感性分析，不能默默归组。
- 非劣界值 δR 与最小有意义节约 Smin：必须在看效果前由临床/业务确定；可展示多个候选阈值的敏感性网格，不把任何示例值称为临床公认标准。

## 2. 原始数据收集清单

| 数据域 | 至少需要的字段 | 用途/局限 |
|---|---|---|
| 患者与保险观察 | 匿名 member_id、产品、参保起止、医疗/药品覆盖、死亡日期及来源 | 去重、可观察性、死亡与删失；死亡源不全需披露 |
| 急性住院 episode | admission/discharge 日期时间、hospital_id、转院标记、出院状态、主次诊断、DRG、手术及时间、ICU、LOS | 生成 index episode，疾病与急性严重度；同一住院多张 claim 必须合并 |
| 出院决策 | SNF/HH 推荐、PT/OT推荐、授权申请/审批/拒绝、原因、时间戳、case management note | 识别 SNF considered 人群；区分决定前信息与决定后过程 |
| 后急性服务 | SNF 入退院、HH 实际访视日期、专业类型、agency/provider_id、服务数量、referral/接单时间 | 治疗分类、两天窗口、服务衔接；不同标识体系先映射 |
| 既往利用 | 30/90/365天 IP/ED/observation、HH/SNF episodes、服务类型、机构、费用 | 混杂调整与预设 HTE；历史窗口全部截于基线 |
| 既往居家成功 | 历史 HH 开始/结束、后续再入院/转SNF、在家存活时间、是否同机构 | 定义成功时需完整历史观察窗；窗口跨到 index 入院之后的历史episode不可用未来信息补齐 |
| 功能和认知 | 入院前/出院决策前 ADL、转移/行走、PT/OT、认知、吞咽、虚弱、辅助器具 | 核心混杂因素；缺失不等于功能良好；服务启动后的 OASIS/MDS 通常不能直接当作治疗前变量 |
| 家庭与环境 | caregiver availability/capacity、独居、楼梯、设备、患者偏好 | 优先结构化或经验证的文本提取；已婚/地区收入不足以替代照护能力 |
| 临床需求 | 氧疗、伤口、注射/输液、管饲、药物负担、肾功能等可得指标及时间 | 识别支持需求与临床资格；仅保留决策时可得版本 |
| 医疗环境 | 地区、月份/年份、医院特征、历史 HH 可及性/质量、SNF 可及性 | 机构和时间混杂；质量指标使用 index 前版本，避免未来信息 |
| 结局事件 | IP/ED/observation 日期、计划性标记、死亡、后续 SNF、功能结果 | 多维健康结果；claims 无法完整代表康复和生活质量 |
| 费用 | medical/pharmacy、paid/allowed、患者分担、服务日期、费用类别、调整/冲销、币值年 | 预先选 payer paid 或 allowed 视角；计入全部可观察服务，不将 HH 低单价视为总节约 |

缺失字段要输出 missingness 和可替代指标；不允许 AI 生成不存在的 caregiver、ADL、死亡或推荐记录。授权拒绝不是天然有效的工具变量。

## 3. 完整结果交付目录

标记：必需=主分析应交付；条件=字段或对应分析存在时必需；增强=不阻塞第一版，但影响结论强度。每个统计表同时保留机器可读数据，所有图有对应源表。

| ID | 阶段/输出 | 必须包含的内容 | 等级 |
|---|---|---|---|
| P01 | 研究协议/estimand 表 | eligibility、A、t0、grace/landmark、follow-up、outcomes、ATO/其他目标、因果假设 | 必需 |
| P02 | 字段/代码字典 | 数据表字段、code list版本、日期含义、单位、可用时间、缺失、计算规则 | 必需 |
| P03 | 分析配置与决策日志 | 窗口、排除顺序、重复episode、模型、阈值、随机种子、确认/待定项 | 必需 |
| D01 | 数据覆盖与质量表 | 年/月/产品样本覆盖、重复claim、异常日期、死亡源、缺失、费用冲销、runout | 必需 |
| D02 | eligibility audit 明细 | 每个episode所有 exclusion flags、首个互斥排除原因、最终纳入状态 | 必需 |
| D03 | Cohort flow 表＋图 | 原始claims→合并住院→患者/episode→顺序排除→HH/SNF/两者/无服务/迟服务→分析样本 | 必需 |
| D04 | 早期窗口事件表 | 启动前/后时间顺序；窗口内死亡、再入院、ED、失保、两者服务；重叠事件单列 | 必需 |
| D05 | 服务启动时间分布 | D0、D1、D2、D3–7及更晚，HH/SNF分别展示；日期精度说明 | 必需 |
| D06 | 纳入/排除人群比较 | 年龄、疾病、虚弱、既往利用、产品/地区；被landmark排除者单列 | 必需 |
| D07 | SNF considered 子队列 | 推荐/申请/审批/实际去向交叉表、时间顺序、缺失与选择性 | 条件 |
| B01 | 原始 Table 1 | HH/SNF unweighted N、均值SD/中位IQR、n%、missingness、SMD | 必需 |
| B02 | 原始结局表 | 每组events/N、风险、观察期、均值费用、分位数、死亡/失保；注明非因果 | 必需 |
| B03 | 基线轨迹/既往护理表 | HH/SNF次数、近期性、成功/失败、同机构、既往IP/ED；变量相关与分布 | 必需 |
| W01 | PS 模型说明 | 仅基线变量、函数形式、交互/非线性、缺失处理、训练方法；不按AUC选PS | 必需 |
| W02 | PS/支持图 | 按治疗组加权前后PS分布、尾部占比、预设亚组局部支持 | 必需 |
| W03 | 权重/ESS 表 | 每组raw N、sum weights、ESS=(sum w)^2/sum(w²)、权重分位数、最大值；复杂设计另列删失权重 | 必需 |
| W04 | 加权 Table 1＋Love plot | 与B01完全同一组变量；OW均值/比例、加权前后SMD、连续变量方差/分布检查 | 必需 |
| W05 | ATO目标人群画像 | 原始eligible vs OW目标分布；哪些患者被降低权重；不可外推群体 | 必需 |
| O01 | 主要健康效果表 | 30天定义明确的再入院：两策略调整风险、ΔR百分点、95%CI；δR、非劣判断 | 必需 |
| O02 | 主要费用效果表 | 90天总费用：两策略调整均值、ΔC、Savings、95%CI；货币年和payer视角 | 必需 |
| O03 | 次要健康效果表 | 死亡、再入院或死亡、ED/observation、60/90天再入院、后续机构使用；功能如可得 | 必需 |
| O04 | 费用分解表/图 | 初始PAC、再入院、ED/obs、后续SNF/HH、门诊、药品等互斥类别；合计核对 | 必需 |
| O05 | 时间趋势图 | 累积再入院风险（死亡竞争风险）、累计费用；标注起点、风险集、删失 | 必需 |
| O06 | 后续照护转移表 | HH→SNF、SNF→HH、重复住院等，不把后续switch作为基线调整项 | 必需 |
| O07 | 健康—费用联合图 | 横轴ΔR、纵轴ΔC，画δR与0费用线，联合bootstrap不确定性 | 必需 |
| H01 | 预设亚组注册表 | 假设、机制、基线定义、方向、优先级、重叠关系、可用字段 | 必需 |
| H02 | 亚组支持/平衡表 | 每亚组两组N/events/ESS、PS范围、加权后SMD、缺失、可估计标记 | 必需 |
| H03 | 亚组效果总表＋forest plot | 每亚组ΔR/ΔC及CI、非劣/节约证据、interaction、multiplicity、estimand | 必需 |
| H04 | 稳定性复核 | 预设重点亚组在独立时期/测试集、替代模型、关键敏感性下结果 | 必需 |
| M01 | 模型数据拆分表 | 患者级train/validation/test、日期、产品、两组样本/事件；患者不交叉 | 必需 |
| M02 | CF实现/泄漏审计 | X/W/Y、nuisance交叉拟合、honesty、参数、support、cluster、版本；信息时间审计 | 必需 |
| M03 | CATE预测分布 | readmission与cost分别的预测分布、分位点、不同seed稳定性；不是个人真实效应 | 必需 |
| M04 | HTE验证表 | 独立测试集按冻结score分箱的DR效应与CI、预测vs估计校准；RATE/TOC可补充 | 必需 |
| M05 | 策略规则注册表 | 安全筛选阈值、节约排序、top-k/容量、clinical gate、support gate、回退策略 | 必需 |
| M06 | top-k独立效果验证表 | 预设5/10/20%等容量的N、ESS、DR ΔR与ΔC及联合不确定性；不只报均值预测 | 必需 |
| M07 | policy value表 | 冻结规则相对同目标人群基准策略的风险、费用、增量、CI；unsupported不估计 | 必需 |
| M08 | 候选人群画像表 | selected vs eligible/剩余人群：基线分布、SMD、enrichment、先验特征一致性 | 必需 |
| M09 | 可解释规则验证 | 开发集总结简洁规则、测试集重新估计效果/覆盖率；画像本身不等于验证 | 增强 |
| S01 | 稳健性矩阵 | 时间窗、队列、PS、重复住院、费用尾部、缺失/失保、未测混杂等改变后的ΔR/ΔC | 必需 |
| S02 | 亚组/模型证据强度表 | discovery/validation、样本和事件充分性、支持、不确定性、是否满足预定标准 | 必需 |
| I01 | 实施与净节约情景表 | 可转介人数、接受/接单/落实率、每人gross saving、增量服务成本、net saving | 条件 |
| I02 | 最终结论与限制表 | 可以支持/不能支持、目标人群、候选画像、早期事件盲区、残余混杂、后续前瞻验证 | 必需 |

### 3.1 表格必须使用的统一列

**主效果表**：analysis_id、cohort_version、time_design、population/estimand、endpoint/window、method、N_HH、N_SNF、ESS_HH、ESS_SNF、events_HH、events_SNF、adjusted_mean_HH、adjusted_mean_SNF、effect、effect_unit、CI_lower、CI_upper、censoring_rule、NI_margin、evidence_status。

**亚组表**：在上表基础上加 subgroup_id/definition、prespecified、support_pass、balance_pass、interaction_estimate/CI/p、multiplicity_method、discovery_or_validation。不要用“一组显著另一组不显著”代替交互检验。

**策略/人群表**：policy_id、selection_rule_version、evaluation_split、reference_policy、target_population、eligible_N、selected_N/percent、selected_N_HH/SNF、selected_ESS、DR_ΔR/CI、DR_ΔC/CI、safety_status、saving_status、clinical_review_status。

**画像表**：feature、selected_distribution、eligible_distribution、nonselected_distribution、SMD、enrichment_ratio、missingness、hypothesis_link、development_or_test。画像可解释筛选特征，不能证明该特征造成收益。

## 4. Cohort flow 和平衡：要回答什么

- flow 同时报告“人”和“住院 episode”，否则重复入院可能造成看似样本量很大。
- 排除计数使用固定顺序的互斥计数以便合计；另保留可重叠 flags 便于审计。早期死亡和再入院可能重叠，不可直接相加。
- 无HH/SNF、D3之后才开始、混合路径，都要出现在全体出院 flow，不能在最开始消失。
- 加权后样本没有凭空变成独立的新患者；sum weights 不是新样本量。raw N、加权总量、ESS 分开报告。
- Table 1 包含年龄、性别、产品、疾病、功能、frailty、index严重度、既往利用和费用、HH历史、家庭支持、医院/地区/时期与缺失情况。先看SMD和分布，不靠样本量驱动的基线p值。
- 可把 |SMD|<0.1 设为常用诊断起点，核心混杂变量要求更严格；阈值不是无混杂的证明。总体平衡不保证亚组平衡。
- 临床缺失可能是系统性的。比较两组缺失率，必要时多重插补（在每个开发/评估流程内正确实施）、缺失机制敏感性分析；不能让完整病例分析默默改变目标人群。

## 5. 主要结局和统计口径

### 5.1 健康

- 主结局建议非计划全因急性再入院，采用预先锁定的代码定义；全因版本作敏感性。固定是否包括同日返回、planned admissions、observation。
- 死亡作为竞争事件处理并单独报告，同时报告“再入院或死亡”复合结局。不能把死亡当成普通无信息删失后宣称HH更安全。
- HH−SNF风险差为主；RR可补充，OR不能被直接叫风险比。非劣判断基于上置信界与预设δR，不基于p>0.05。
- 功能恢复、患者体验、照护者负担如无数据，写明无法验证；readmission不变不等于整体health outcome不变。
- healthy days at home受照护地点定义机械影响，只能作辅助，不能单独作为HH更健康的证据。

### 5.2 费用

- 主费用建议出院后90天总payer paid或allowed（择一）；只看SNF/HH费用不足以证明总节约。若landmark版本主费用从landmark开始，应另列0至landmark已发生费用，不能混称完整出院策略效果。
- 分析算术均值差，因为预算关心总额；中位数/IQR与尾部分布辅助报告。不要只报告log-cost差或删去所有高费用患者。
- index admission费用在出院时已发生，通常作为基线或补充episode总额，不能称为出院后替代带来的节约。
- 对重叠结算、调整冲销、跨窗服务的费用分配设一致规则。后续crossover费用继续纳入初始路径分析。
- 死亡后实际费用可为零但必须与死亡结局共同解释；不以仅存活者费用作为默认主结论。失保后的未知费用不得置零。
- 有删失时选择与estimand匹配的费用估计/观察概率调整。bootstrap保留患者相关性并在需要时重估PS/结局模型；医院级相关的处理另行预设。

### 5.3 联合成功标准

主候选组需同时具有健康非劣证据和总费用节约证据。若预设使用95%双侧CI，则可采用 UCL(ΔR)<δR 且 UCL(ΔC)<0（或<−Smin）作为相应证据标准；CI等级与单侧alpha在协议固定。CI跨界则标记“证据不足”，不能自动判为安全或无效。

多亚组/多策略筛选应控制选择乐观与多重比较。用成对bootstrap保留费用和健康之间的相关性；不把两个独立模型的置信概率相乘来宣称联合安全概率。

## 6. 第二步：预设亚组

| 亚组 | 推荐基线定义 | 注意事项 |
|---|---|---|
| 既往成功HH | 过去固定窗口内HH后一定时段无IP/转SNF，且历史结局已完全在index前观察到 | 无HH、成功HH、失败HH分开；次数多不等于成功 |
| 护理关系连续 | 历史agency集中度、最近是否已有active HH关系 | 本次实际同agency复接是治疗实现变量，不能当普通基线分组；可研究预先已确认的接续方案 |
| 康复主导疾病 | 择期无并发症关节置换单列；骨折、卒中、内科住院分开 | 不把所有骨科混在一起 |
| 功能损失小 | 基线至出院决策前ADL/转移/步行变化 | claims代理不足时明确降级解释 |
| 家庭支持 | 决策前已确认的帮助能力与可用性 | 不用已婚直接代表充分照护 |
| 认知×家庭支持 | dementia程度、行为症状与支持组合 | 多重交互仅探索，受样本/事件限制 |
| 服务可及性 | 出院前可得的历史接单/及时首访能力、地区服务供给 | 本次实际首访时间/次数是后续过程，另作策略研究 |
| 历史稳定度 | 近期IP/ED次数、距上次入院、frailty、近期费用轨迹 | 同时是混杂因素与候选效应修饰变量 |

执行建议：主方案用整体PS模型加入预设关键交互/非线性，在亚组内检查并汇总同一套overlap目标权重下的效果。若需亚组重拟合PS，重新报告其目标人群；不同亚组OW可能对应不同分布。交互比较在明确的共同建模/标准化框架下做。样本不足、两组无支持或ESS过小的亚组只描述，不强估效果。PSM复核若目标变为ATT，需标注，不能把差异简单叫“不稳健”。

## 7. 第三步：causal forest 和策略验证

### 7.1 模型设计

- 使用相同cohort定义和治疗/结局时间口径，但不必使用PSM删减后的数据；局部无重叠部分不做强外推。
- 两个主要CATE模型：τR(X)=E[R(HH)−R(SNF)|X]；τC(X)=E[C(HH)−C(SNF)|X]。二元readmission对应风险差，不是个人会不会发生事件。
- X只能是部署决策时可得信息。服务后的OASIS/MDS、实际HH次数、SNF住院天数、后续费用均不得泄漏。
- 使用honest/orthogonalized实现，nuisance模型交叉拟合。不要因主分析用OW就机械地把同一治疗权重再次塞进forest；按所选实现定义训练目标和最终加权评价目标。
- CF分箱效果和总体OW结果应在同一目标分布/时间口径下比较。若CF面向未经OW的eligible人群，明确其与ATO不同。
- 患者级split；时间外验证如跨期患者重复出现需预定处理，主独立测试不允许同一患者泄漏。样本量不足时采用嵌套交叉拟合，仍不得边看测试结果边调规则。

### 7.2 选择规则

开发集内先按临床资格、support、预测τR的预设阈值定义候选，再按−τC排序。预测τR≤δR只是筛选条件，不是已证实安全。选择top-k阈值后冻结，在独立测试集用DR方法估计候选群的平均ΔR/ΔC及CI。每个人不必都有精确的个体CI，但群体安全/节约证据不能省。

“前10%”分母必须明确：所有eligible、通过安全预测门槛者，还是可考虑SNF替代者。不得三个分母混用。阈值/容量可按业务预先设多个并完整报告，不能只展示最有利的一档。

### 7.3 验证层次

1. **效应异质性**：冻结预测分数后，测试集分箱DR效应、CI、校准；RATE/TOC检验排序能力。普通再入院预测AUC不是CATE质量证据。
2. **联合选人**：在相同selected组中同时验证风险差和费用差，保留联合不确定性，不能用不同人群分别证明安全与省钱。
3. **策略价值**：冻结π(X)规定HH/SNF，在相同目标人群内用off-policy/DR评价，与临床可行且有支持的参考策略比较。可评估全SNF/全HH只限两策略均有支持且有临床意义的目标人群；“当前实际分配”的观察均值不是自动可交换的策略对照。
4. **画像解释**：selected组富集某些先验特征仅表示一致性；新规则需再验证。feature importance/SHAP不能证明某特征导致治疗收益。

锁定策略的测试CI只反映该规则的评价不确定性；若需要覆盖整个学习/选择流程的不确定性，另用重复split/嵌套bootstrap并说明计算与推断方法。

## 8. 敏感性与失败结果也必须交付

| 改变项 | 要解决的问题 |
|---|---|
| 时间设计/窗口：计划分组、landmark、grace-period；D1/D2/D3 | 早期事件选择、服务延迟及不同estimand；不可把不同问题结果直接混合 |
| 双重服务归组、长期机构居民、hospice、转院合并 | treatment和eligibility误分类 |
| first episode vs all episodes | 重复事件、窗口重叠与聚类 |
| PS设定、OW vs有明确目标的PSM/其他估计 | 模型依赖；目标人群差异与估计差异分开 |
| 亚组局部平衡/overlap、保险产品/疾病/医院/时期 | positivity与可迁移性 |
| 未测混杂分析 | caregiver/ADL/偏好缺失可造成多大偏差；使用适合目标尺度的定量敏感性，不能声称已消除 |
| 费用高尾、paid vs allowed、60 vs90天 | 节约是否由少数病例或口径驱动；原始均值保留主位 |
| 失保、死亡源、缺失插补 | 不完整随访和信息偏差 |
| 多split/时间外测试、阈值网格 | HTE与策略选择是否稳定 |

若结果未通过，应输出：缺乏支持、残余失衡、事件不足、CI过宽、非劣未证实、费用节约未证实、外部验证失败。不要反复删人/调窗口直至“成功”。

## 9. 最终报告的主文顺序

1. 研究问题和时间轴（P01）。
2. 人群流转（D03）及早期事件局限（D04）。
3. Table 1加权前后（B01/W04）、PS与ESS（W02/W03）。
4. 主效果及费用分解（O01/O02/O04），健康—费用联合图（O07）。
5. 预设亚组forest plot（H03），明确发现/确认性质。
6. CF独立验证与top-k收益/风险（M04/M06/M07）。
7. 最终画像（M08）、适用范围、实施净节约与限制（I01/I02）。

完整审计、原始结局、代码字典、模型参数、全部敏感性放附录。主文始终回答：这是谁的效果、对比什么策略、从何时计时、差多少、证据有多确定。

## 10. 可直接交给 Claude / Codex 的执行说明

> 请按本文件P01–I02实现SNF vs HH分析流水线。先检查实际数据字典，建立字段映射与缺失能力矩阵，不猜字段、不生成患者结果。所有窗口、代码集、treatment、time zero、estimand、exclusion和NI界值放进配置。尚未确定的设计以pending标记；不阻塞已可完成的数据覆盖、cohort审计和描述性输出。
>
> 按“协议→数据QA→cohort→基线/权重→总体效果→预设亚组→CF开发→冻结策略独立测试→画像/业务情景”顺序执行。每阶段交付对应ID的源数据表、图表和结果说明；记录输入版本、代码版本、行数、患者数、配置hash和随机种子。没有支持或验证未通过时保留失败结果，不静默改设计。
>
> 主分析默认PS→OW，不串联PSM；二者若同时做，分别标注目标人群。仅用决策时可得变量。按member拆分训练和测试。最终测试集不能用于特征选择、阈值调参或反复选亚组。输出每个selected组的独立DR平均风险差与费用差、CI、ESS和局部支持，不只交付CATE预测均值。
>
> 先完成P01/P02/P03及D01–D07；依据缺失字段明确哪些问题可回答，再继续估计。运行过程中所有真实患者数据应留在用户授权的数据环境，报告只输出允许的汇总统计。

建议数据产品：episode_master、eligibility_audit、baseline_features、treatment_timeline、outcomes_costs、analysis_weights、split_manifest、cate_predictions、policy_evaluation、result_manifest。主键统一member_id+episode_id；时间序列表另加service/event时间。结果manifest记录output_id、路径、cohort版本、时间设计、目标人群、方法、endpoint、split、状态、限制。

### 验收门槛

- flow逐步人数可核对，排除总数与原始episode对应；费用类别加总一致。
- t0/landmark/随访窗口与每张结果表一致；早期事件未被隐藏。
- treatment互斥规则与baseline信息时间可审计。
- 两组支持和关键混杂变量平衡达到预设诊断要求；否则不出确定性因果结论。
- 全部选人规则和NI界值在测试前冻结；测试无患者泄漏。
- “节约且健康不劣”必须在同一个人群、同一目标策略比较下联合成立；不确定时明确写证据不足。

## 11. 方法来源

这些文献支持方法选择，不是该研究尚未分析的数据结论。

1. Hernán et al. (2016). Specifying a target trial prevents immortal time bias and other self-inflicted injuries in observational analyses. https://pubmed.ncbi.nlm.nih.gov/27237061/ （time zero与grace period）
2. Li, Thomas & Li (2019). Addressing Extreme Propensity Scores via the Overlap Weights. https://pubmed.ncbi.nlm.nih.gov/30189042/ （OW与重叠目标人群）
3. GRF官方：average_treatment_effect. https://grf-labs.github.io/grf/reference/average_treatment_effect.html （DR汇总效果）
4. GRF官方：rank_average_treatment_effect. https://grf-labs.github.io/grf/reference/rank_average_treatment_effect.html （独立排序验证、RATE/TOC）
5. GRF官方指南. https://grf-labs.github.io/grf/articles/grf_guide.html （CATE、异质性及策略评价）

