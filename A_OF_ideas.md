可以，而且我搜完之后觉得，你现在这个思路其实可以“拔高”得相当多。它不应该被描述成：

> “我把以前经常 query 的 member history 预先存成一张表，方便以后查。”

更准确的定位应该是：

**把原本 transaction-oriented 的 claims / encounter / authorization 数据，转化成一个 member-centric、longitudinal、event-based 的统一表征，然后把它作为 Care Trajectory / Patient Journey Discovery 的基础数据层，用于系统化 opportunity finding。**

换句话说，你做的已经不只是 EDA，而是在搭一个 **Longitudinal Member Journey Analytics Foundation**。

---

## 一、你的想法其实正好对应了一个很成熟的学术脉络

最核心的理论不是普通的 EDA，而是 **Care Trajectory / Patient Journey + Process Mining + Sequence Analysis**。

Process mining 的基本思想就是：不要把医疗数据只看成“某人有几个 claim、几个 admission”，而要把它重新表达成：

**Case → Event → Time → Sequence → Pathway**

也就是：

`Member → ER → IP admission → discharge → SNF auth request → denial → resubmission → approval → SNF stay → readmission`

Process mining 最基本的 event log 就只要求 `Case ID + Activity + Timestamp`，然后再往上增加 provider、cost、resource、diagnosis、outcome 等属性。医疗领域已经大量采用这种表示。([PubMed Central (PMC)][1])

而且 2022 年的 *Process Mining Handbook* 已经把 healthcare 作为专门的一章，因为医疗流程具有典型的“loosely structured、knowledge-intensive、highly variable”特征，非常适合从实际事件记录中反推真实 pathway。([Springer Link][2])

所以你原来的：

**“每次有问题 → 临时写 SQL → pull 这个 member 的 history”**

实际上是在反复做：

> **on-demand reconstruction of an event log**

你现在想做的则是：

> **persistent standardized longitudinal event representation**

这是性质完全不同的一件事。

---

# 二、我觉得最适合你这件事情的概念，是「Care Trajectory」

这里甚至有一个和你的设计几乎一模一样的理论模型：**6W model of Care Trajectories**。

2018 年 *Public Health* 的一篇综述把 care trajectory 拆成：

| 维度        | 理论文献定义          | 放到你数据里是什么                                  |
| --------- | --------------- | ------------------------------------------ |
| **Who**   | 谁在接受 care       | member、年龄、risk、comorbidity、历史 utilization  |
| **Why**   | 为什么进入医疗系统       | diagnosis、acute condition、admission reason |
| **Which** | 与哪些 provider 互动 | hospital、physician、SNF、HHA                 |
| **Where** | 在哪里接受 care      | ER、IP、SNF、home、outpatient                  |
| **What**  | 接受了什么 service   | admission、HHS、SNF、procedure、authorization  |
| **When**  | 时间及事件顺序         | start/end、lag、transition、sequence          |

这个模型本身就是为了把患者在医疗系统中的一系列 interaction 从零散事件提升成 **trajectory**；而且作者特别指出，它可以被用于发现降低 readmission、改善 care 的 **“missed opportunities”**。这与你所谓 opportunity finding 的目标非常接近。([ScienceDirect][3])

后续研究已经真的把它用在 administrative healthcare data 上，通过 State Sequence Analysis 把患者变成一条条 utilization trajectory，比如：

**hospital → ED → home → PCP → specialist → hospital**

然后比较不同 trajectory 类型的 patient characteristics 和 outcome。([Springer Link][4])

所以，你以后完全可以把自己这个工作描述成：

**Longitudinal Care Trajectory Reconstruction and Opportunity Discovery**

而不是单纯叫 EDA。

---

# 三、而你想到的“先做成表”其实还有一个专门的方法学名字：Event Abstraction

这点我觉得特别重要。

Claims 数据最底层往往是：

claim line、procedure line、facility claim、professional claim、authorization status change……

这些并不是你真正希望分析的 clinical/business event。

例如：

```text
8 claim lines
3 diagnosis codes
4 procedure codes
2 dates
```

对你来说真正有意义的可能只是：

```text
2026-01-03    ER visit
2026-01-03    Acute inpatient admission
2026-01-08    IP discharge
2026-01-08    SNF authorization submitted
2026-01-09    SNF authorization denied
2026-01-10    Home health initiated
2026-01-18    ED revisit
```

这个从 **low-level transactional records → clinically meaningful events** 的过程，在 process-mining 文献中就叫 **event abstraction**。

已有综述明确指出，现实世界中的信息系统记录往往处于不同 granularity，所以必须先把细粒度 event 转换到适合分析的 coarse-grained event，process mining 才真正有意义。([Springer Link][5])

Healthcare-specific 的工作也特别强调这一点：不同医疗数据库的 granularity 不同，event abstraction 是有效构造医疗 event log 的重要环节。([Icahn School of Medicine at Mount Sinai][6])

所以你这个工作里非常有价值的一部分实际上是：

**Healthcare Event Ontology / Event Abstraction Layer**

而不是 ETL。

---

# 四、所以我反而不建议你真的只做“一张 member 一行的大宽表”

这是我看完这些文献以后对你设计上最重要的建议。

真正长期有价值的核心应该有三个逻辑层：

| 层                            | Grain                          | 用途                                  |
| ---------------------------- | ------------------------------ | ----------------------------------- |
| **Canonical Event Layer**    | 1 row = 1 meaningful event     | 保留完整 journey                        |
| **Episode / Journey Layer**  | 1 row = 1 episode / transition | IP episode、SNF episode、auth episode |
| **Feature / Snapshot Layer** | 1 row = member × index date    | PSM、OW、HTE、prediction               |

例如你的核心 event 表可以长这样：

```text
member_id
event_id
event_domain
event_type
event_subtype

event_start
event_end

episode_id
authorization_id

setting
provider_type

diagnosis_group
service_type

auth_status
cost

source_table
source_record_id
```

其中：

`event_domain`

甚至可以专门区分：

```text
UTILIZATION
AUTHORIZATION
CARE_MANAGEMENT
SERVICE
OUTCOME
```

这样一个 member 的 journey 才真正可以组合起来。

这与现有 process-mining event-log architecture 是一致的：最小结构是 case/activity/timestamp，但实践中会增加属性来支持 performance、cost、resource、outcome 等分析。([PLOS][7])

---

# 五、你这里甚至比普通 Patient Journey 更复杂：你非常适合 Object-Centric Process Mining

这个是我这轮搜索里觉得**尤其适合你**的一个新方向。

传统 patient journey 默认：

> Case = member

但你的现实情况不是这么简单。

比如一个 member 同时有：

```text
Member A
 ├── IP Episode 1
 │     ├── Auth Request #123
 │     │      ├── submitted
 │     │      ├── denied
 │     │      └── approved
 │     └── SNF Stay #56
 │
 ├── IP Episode 2
 │     └── HHS Episode #89
 │
 └── Care Management Episode #12
```

也就是说，一个 event 可能同时属于：

**member + admission episode + authorization + service episode**

这正是 traditional process mining 的一个限制。

2024 年 *Journal of Biomedical Informatics* 专门发表了 healthcare **Object-Centric Process Mining (OCPM)** 的方法论文，就是为了解决这种问题：不再强迫所有事件只有一个 patient/case ID，而允许 healthcare event 同时关联多个 object。([ScienceDirect][8])

这对你的 **pre-auth → deny → approve → actual utilization** 尤其有意义。

所以你的第一版不必真的实现完整 OCPM，但表结构最好已经保留：

`member_id`

`episode_id`

`authorization_id`

`service_episode_id`

未来就不会被 member-only journey 锁死。

---

# 六、然后它就可以真正变成你的「Opportunity Finding Engine」

这里是你现在这个工作最值得拔高的一步。

你以前的 workflow 更接近：

**Hypothesis → Query → Cohort → OW → Outcome**

比如：

> SNF vs HHS 有没有机会？

这是 **hypothesis-driven opportunity finding**。

有了 Member Journey Event Store 以后，你就可以增加第二条路线：

**Data → Journey → Pattern → Opportunity Hypothesis → Causal Validation**

也就是说：

```text
Raw Claims / Auth / CM
          ↓
Event Abstraction
          ↓
Canonical Member Journey
          ↓
Trajectory / Sequence Discovery
          ↓
Abnormal / costly / high-risk pathway
          ↓
Opportunity Hypothesis
          ↓
Causal Analysis
          ↓
Actionable Intervention
```

这其实非常重要。

因为 **journey mining 负责 hypothesis generation**；

而你已经在做的 **PSM / overlap weighting / HTE / causal forest** 负责 hypothesis validation。

两者接起来之后，你整个 opportunity-finding framework 就完整了。

---

## 七、而且已经有很多正式方法可以直接建立在这张表上

2023 年 *BMC Medical Research Methodology* 做了一篇很好的 scoping review，系统审查了 healthcare utilization sequence 的分析方法。51 项研究里，大致可以分成两类：

**完整 trajectory** 通常用 clustering / sequence analysis；

**局部 subsequence** 通常用 pattern mining / Markov 等方法。([PubMed Central (PMC)][9])

因此你以后可以很自然地沿着这一条 analysis ladder 往上走：

1. **Individual journey reconstruction**
   “这个 member 到底经历了什么？”

2. **Transition analysis**
   `IP → SNF`、`IP → HHS`、`ER → IP` 的概率是多少？

3. **Timing / delay analysis**
   discharge 后多久开始 HHS？
   auth denied 到 approved 隔多久？

4. **Frequent sequence mining**
   哪些 sequence 特别常见？

5. **Trajectory clustering / State Sequence Analysis**
   自动找出几种典型 member journey。

6. **Process discovery**
   从 data 自动生成实际 care pathway。

7. **Conformance / deviation analysis**
   哪些人偏离理想路径？
   哪里发生 delayed、skipped、repeated event？

8. **Outcome-linked pathway analysis**
   哪种 pathway 对应更高 readmission / cost？

9. **Opportunity identification**
   哪个 transition 是可干预的？

10. **Causal validation**
    OW / DR / HTE / causal forest 验证如果改变 transition 是否真的改善 outcome。

Process mining 本身就通常分成 **discovery、conformance checking、enhancement**；最近 healthcare 系统综述也仍然把这些作为主要框架。([PubMed Central (PMC)][10])

---

# 八、你的 auth 数据反而可能是这套东西里非常有价值的部分

因为很多 patient journey 研究只有：

> “患者实际上去了哪里。”

而你的数据可能同时告诉你：

> **what was intended → what was authorized → what actually happened**

比如：

```text
IP discharge
 ↓
SNF requested
 ↓
SNF denied
 ↓
HHS initiated
 ↓
30-day outcome
```

和：

```text
IP discharge
 ↓
SNF requested
 ↓
SNF approved
 ↓
SNF admission
 ↓
30-day outcome
```

这已经不仅仅是 utilization trajectory。

它实际上是：

**Decision → Care Pathway → Outcome**

这里面就可能产生非常好的 natural variation / quasi-experimental hypothesis。

例如你现在的 SNF downgrade 研究，本质上就可以由 journey layer 自动识别：

```text
Intended SNF
       ↓
 ┌───────────────┐
SNF actual      HHS actual
 ↓                ↓
cost           cost
readmission    readmission
```

再进一步：

```text
prior HHS-heavy history
             ×
SNF → HHS transition
             ↓
    heterogeneous outcome
```

这就和你现在的 HTE 分析完全接上了。

---

# 九、Temporal Phenotyping 也是另一条理论支撑

这里还有一个很重要的思想：

**医疗信息不只是“发生过什么”，event 的顺序本身就是 information。**

JAMIA 的 temporal phenotyping 工作就发现，把 diagnosis/medication 当作 temporal sequences，而不是简单聚合成 counts，可以得到额外的 phenotype 信息；其核心动机就是传统 EHR feature 往往丢掉 temporal structure。([PubMed][11])

对你同样适用。

比如：

```text
3 IP
2 ER
1 SNF
```

与：

```text
ER → IP → SNF → ER → IP → SNF
```

在普通 aggregated feature table 里可能非常相似。

但从 care management 角度，它们根本不是一个人群。

因此你的 journey table 真正保存的是一种：

> **temporal representation of healthcare behavior**

这也是为什么我不建议只生成 aggregated member table。

---

# 十、再往最高一层拔，它其实是 Learning Health System 的一个小型基础设施

这个我觉得你以后给 leadership 解释这个项目时特别好用。

Learning Health System 的基本循环就是：

**Practice → Data → Knowledge → Improved Practice**

BMJ 2024 对 LHS 的总结就是，routine healthcare event 应该被系统记录、整合、分析，然后重新反馈到 healthcare improvement 中；data warehouse / registry 等 reusable data infrastructure 是这个循环的技术基础。([BMJ][12])

系统综述也把 LHS 描述成：

> routine healthcare data → knowledge generation → iterative healthcare improvement

并特别强调 longitudinal patient tracking。([PubMed Central (PMC)][13])

所以你的项目最高层其实可以描述成：

**把 care delivery 的 operational data 转化成 reusable longitudinal knowledge substrate，从而支持持续的 care-management opportunity discovery。**

这就比：

> “建个 table 少写 SQL”

高了至少两个层级。

---

# 十一、如果让我给你现在这个项目重新命名，我会这样

数据资产本身，我最推荐：

**Longitudinal Member Journey Event Store**

或者更业务化一点：

**Member Journey Data Mart**

分析方法：

**Care Trajectory Analytics**

整个 opportunity 项目：

**Journey-Based Care Opportunity Discovery**

如果写到 proposal / presentation 里，我会用一句：

> **Build a reusable longitudinal member-journey data foundation that transforms fragmented claims, utilization, and authorization records into standardized care events, enabling systematic pathway discovery, trajectory segmentation, deviation detection, and downstream causal evaluation of care-management opportunities.**

这句话基本完整表达了你真正做的事情。

---

## 十二、我会把你整个 Opportunity Finding 方法论最终整理成这个框架

你现在其实已经逐渐形成一套很不错的结构：

**Layer 1 — Journey Representation**

Claims + utilization + auth + CM
→ standardized longitudinal event representation

**Layer 2 — Exploratory Opportunity Discovery**

process discovery

* sequence mining
* trajectory clustering
* transition / timing analysis
* abnormal pathway detection

**Layer 3 — Outcome Association**

pathway / subgroup
→ readmission / cost / utilization

**Layer 4 — Causal Validation**

PSM / Overlap Weighting / DR
→ average opportunity

HTE / subgroup / causal forest
→ targeted opportunity

**Layer 5 — Intervention Translation**

“哪个 member”

* “journey 到哪个节点”
* “发生什么 signal”
  → care manager 应采取什么 action

所以，**你现在这个 Member Journey 表其实不是一个旁支 EDA 项目，而应该成为你整个 opportunity-finding framework 的最底层 infrastructure。**

而且我认为这里最重要的设计原则是：

> **不要一开始就把 journey 压扁成 features；先保留 event sequence，再按具体研究动态 derive features。**

这是你这一步最值得保住的东西。

几篇我认为你应该重点留着的理论支撑是：6W Care Trajectory 模型 [Public Health — 6W multidimensional model](https://doi.org/10.1016/j.puhe.2018.01.007?utm_source=chatgpt.com)；healthcare utilization sequence 方法综述 [BMC — Analytical methods for sequences of utilization](https://link.springer.com/article/10.1186/s12874-023-02019-y?utm_source=chatgpt.com)；OMOP → process-mining event log [PLOS ONE — OMOP for healthcare process mining](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0279641&utm_source=chatgpt.com)；Object-Centric healthcare process mining [JBI — Object-centric process mining in healthcare](https://www.sciencedirect.com/science/article/pii/S153204642400100X?utm_source=chatgpt.com)；以及最近 2026 年用 administrative data 找 care-pathway variation、delay、duplication 和 cost difference 的实际研究。([PLOS][14])

**如果把这个和你前面那套 OW → subgroup HTE → causal forest 的 Opportunity Finding workflow 合起来，其实已经可以形成一套相当完整的“Journey Discovery → Causal Opportunity Validation”方法学。**这个是我觉得你目前整个工作里很值得正式系统化的一条主线。

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7557979/?utm_source=chatgpt.com "Mapping the Patient’s Journey in Healthcare through Process Mining - PMC"
[2]: https://link.springer.com/chapter/10.1007/978-3-031-08848-3_14?utm_source=chatgpt.com "Using Process Mining in Healthcare | Springer Nature Link"
[3]: https://www.sciencedirect.com/science/article/pii/S0033350618300106?utm_source=chatgpt.com "The ‘6W’ multidimensional model of care trajectories for patients with chronic ambulatory care sensitive conditions and hospital readmissions - ScienceDirect"
[4]: https://link.springer.com/article/10.1186/s12913-020-5030-0?utm_source=chatgpt.com "Healthcare utilization after a first hospitalization for COPD: a new approach of State Sequence Analysis based on the '6W' multidimensional model of care trajectories | BMC Health Services Research | Springer Nature Link"
[5]: https://link.springer.com/article/10.1007/s41066-020-00226-2?utm_source=chatgpt.com "Event abstraction in process mining: literature review and taxonomy | Granular Computing | Springer Nature Link"
[6]: https://scholars.mssm.edu/en/publications/domain-specific-event-abstraction-2?utm_source=chatgpt.com "Domain-specific Event Abstraction - Icahn School of Medicine at Mount Sinai"
[7]: https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0000647&utm_source=chatgpt.com "Utilizing process mining in quality management: A case study in radiation oncology | PLOS Digital Health"
[8]: https://www.sciencedirect.com/science/article/pii/S153204642400100X?utm_source=chatgpt.com "Enhancing healthcare process analysis through object-centric process mining: Transforming OMOP common data models into object-centric event logs - ScienceDirect"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10523647/?utm_source=chatgpt.com "Analytical methods for identifying sequences of utilization in health data: a scoping review - PMC"
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11784703/?utm_source=chatgpt.com "Process mining applications in healthcare: a systematic literature review - PMC"
[11]: https://pubmed.ncbi.nlm.nih.gov/33313899/?utm_source=chatgpt.com "High-throughput phenotyping with temporal sequences - PubMed"
[12]: https://www.bmj.com/content/384/bmj-2023-076175?utm_source=chatgpt.com "Data to knowledge to improvement: creating the learning health system | The BMJ"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7932903/?utm_source=chatgpt.com "Learning health systems using data to drive healthcare improvement and impact: a systematic review - PMC"
[14]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0339788&utm_source=chatgpt.com "Enhancing quality and decision-making for care pathways: An application of process mining in cancer care | PLOS One"
