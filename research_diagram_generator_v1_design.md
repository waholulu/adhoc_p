# Research Diagram Generator v1 设计方案

## 0. 设计目标

这个方案的目标不是让 Claude Code “会画图”，而是让它能把复杂的医疗数据研究设计、SQL 逻辑、分析流程和结果解释，转成适合经理、总监和跨职能团队理解的视觉材料。

核心目标：

1. 把研究设定讲清楚，包括 cohort、index date、baseline window、exposure window、outcome window、cost window。
2. 把数据逻辑讲清楚，包括 claims、enrollment、auth、care management、provider、SDOH 等数据源如何进入 final analytic table。
3. 把方法逻辑讲清楚，包括 adjustment、matching、weighting、causal forest、policy tree、validation 和 sensitivity analysis。
4. 输出比 Mermaid 更美观、更适合汇报的 HTML 加 inline SVG 图。
5. 保持轻量，不一开始做复杂多 agent 系统。

一句话设计：

短命令负责触发，reference 负责表达规则，Skill 负责 HTML/SVG 渲染，模板负责稳定美观。


## 1. 总体架构

推荐目录结构：

```text
.claude/
  commands/
    visual-brief.md
    study-diagram.md
    diagram-review.md

  references/
    visualization/
      index.md
      visual_style_guide.md
      research_visual_templates.md
      chart_rules.md
      diagram_antipatterns.md

  skills/
    research-diagram/
      SKILL.md
      assets/
        base_template.html
        study_design_template.html
        cohort_flow_template.html
        data_lineage_template.html
        causal_framework_template.html
        analysis_pipeline_template.html
        result_summary_template.html
```

第一阶段最小可落地版本：

```text
.claude/
  commands/
    visual-brief.md
    diagram-review.md

  references/
    visualization/
      visual_style_guide.md
      research_visual_templates.md
      chart_rules.md

  skills/
    research-diagram/
      SKILL.md
      assets/
        base_template.html
```

不建议第一阶段就做：

```text
复杂多 agent team
自动读取所有 SQL 和所有文档
自动生成完整 PowerPoint
自动运行分析并画结果
复杂 JavaScript 交互图
```

第一阶段先把“研究逻辑到漂亮静态图”做好。


## 2. 核心分工

| 组件 | 负责什么 | 是否第一阶段需要 |
|---|---|---|
| `visual-brief.md` | 把研究方案变成一组适合汇报的视觉建议和图规格 | 是 |
| `study-diagram.md` | 生成单张 HTML/SVG 研究图 | 可选，第二步加 |
| `diagram-review.md` | 审核图是否清晰、美观、准确、适合领导看 | 是 |
| `visual_style_guide.md` | 固定视觉语言、配色、字体、布局规则 | 是 |
| `research_visual_templates.md` | 规定不同研究图怎么组织信息 | 是 |
| `chart_rules.md` | 规定结果图和统计图怎么选 | 是 |
| `diagram_antipatterns.md` | 收集糟糕图形模式，防止反复犯错 | 可选 |
| `research-diagram/SKILL.md` | 负责按模板生成 standalone HTML + inline SVG | 是 |
| `base_template.html` | 固定 HTML 页面壳和 SVG 样式 | 是 |


## 3. 为什么不用 Mermaid 作为主输出

Mermaid 适合快速表达逻辑，但经常有几个问题：

1. 默认观感不够 polished。
2. 布局控制有限。
3. 信息稍微复杂就容易拥挤。
4. 很难加入精致的 summary cards、callout、legend 和品牌化样式。
5. 不适合直接当作领导汇报图的最终形态。

本方案采用：

```text
结构化研究信息
→ 图规格
→ HTML + inline SVG
→ 浏览器预览
→ 截图、导出 PDF、或放入 PPT
```

这样能同时获得可控布局和更美观的呈现。


## 4. 输出标准

每一张研究图应该是一个 standalone HTML 文件。

要求：

1. 单个 `.html` 文件。
2. 内嵌 CSS。
3. 内联 SVG。
4. 不依赖外部图片。
5. 不需要 JavaScript。
6. 可以浏览器直接打开。
7. 可以截图进 PowerPoint。
8. 可以打印或导出 PDF。
9. 不包含 PHI、PII 或真实 row-level identifier。

推荐文件命名：

```text
outputs/
  hhs_48h_study_design.html
  readmission_cohort_flow.html
  post_acute_data_lineage.html
  cost_outcome_analysis_pipeline.html
```


## 5. 视觉设计系统

### 5.1 两套主题

建议保留两套主题。

#### Light Executive Theme

适合：

```text
领导汇报
PowerPoint
打印
跨部门讨论
正式项目文档
```

推荐样式：

```text
背景：#F8FAFC
主文字：#0F172A
次级文字：#475569
卡片背景：#FFFFFF
卡片边框：#CBD5E1
弱背景：#F1F5F9
```

#### Dark Technical Theme

适合：

```text
技术文档
内部方案图
工程感较强的数据 pipeline 图
Claude Code 生成预览
```

推荐样式：

```text
背景：#020617
主文字：#E5E7EB
次级文字：#94A3B8
卡片背景：#0F172A
卡片边框：#334155
弱背景：#111827
```

第一阶段建议默认使用 Light Executive Theme，因为你的主要受众常常是 manager 和 director。


### 5.2 语义配色

固定颜色含义，让同类内容每次都用同一类颜色。

| 元素类型 | 用途 | Light Theme 建议 | Dark Theme 建议 |
|---|---|---|---|
| Data Source | claims、enrollment、auth、CM、provider | 蓝色 | 蓝色 |
| Cohort | population、eligibility、exclusion | 青色 | 青色 |
| Exposure | HHS、SNF、intervention、treatment | 绿色 | 绿色 |
| Outcome | readmission、cost、utilization | 紫色 | 紫色 |
| Method | weighting、matching、model、causal forest | 黄色 | 黄色 |
| Confounder | acuity、prior utilization、SDOH | 橙色 | 橙色 |
| Risk or Limitation | leakage、bias、missingness | 红色 | 红色 |
| External Context | market、facility access、policy | 灰色 | 灰色 |

建议颜色：

```text
Data Source: #2563EB
Cohort: #0891B2
Exposure: #16A34A
Outcome: #7C3AED
Method: #CA8A04
Confounder: #EA580C
Risk: #DC2626
External: #64748B
```


### 5.3 字体

建议：

```text
领导汇报版：Inter、Aptos、Segoe UI、Arial
技术文档版：Inter、JetBrains Mono for small code labels
```

规则：

1. 标题用 22 到 28 px。
2. 副标题用 13 到 15 px。
3. 节点标题用 13 到 15 px。
4. 节点副标签用 10 到 12 px。
5. 注释用 10 到 12 px。
6. 不要用太多 monospace，除非是表名、字段名、SQL 片段。


### 5.4 布局规则

必须遵守：

1. 一个主图最多 9 个主节点。
2. 超过 9 个主节点就拆成多张图。
3. 主流程从左到右或从上到下，不要混用。
4. 每个节点最多 2 行主文字。
5. 复杂注释放在 summary cards，不要塞进主图。
6. 箭头不能穿过文字。
7. Legend 放在主图底部或右下角，不能压住节点。
8. 一个图只讲一件主要事情。
9. 边界框内至少留 24 px 内边距。
10. 节点之间最少保留 32 px 间距。


## 6. 固定页面结构

每个 HTML 图使用统一结构：

```text
Header
Main Diagram
Summary Cards
Assumptions and Limitations
Footer
```

### 6.1 Header

包含：

1. 图标题。
2. 一句话副标题。
3. 可选标签，如 Study Design、Cohort Flow、Data Lineage。

示例：

```text
Title:
Early HHS After Discharge Study Design

Subtitle:
Index discharge anchored design comparing HHS within 48 hours versus delayed or no HHS
```

### 6.2 Main Diagram

主图只展示最重要的结构。

例如：

```text
Baseline window
→ Index discharge
→ Exposure window
→ Outcome window
→ Cost window
```

### 6.3 Summary Cards

固定 3 张卡片。

推荐卡片类型：

```text
Business Question
Design Definition
Key Risks
```

或者：

```text
Cohort
Outcome
Validation
```

规则：

1. 每张卡最多 4 个 bullet。
2. 每个 bullet 不超过 14 个英文词或 18 个中文字符。
3. 卡片内容用于补充主图，不重复主图全部信息。

### 6.4 Assumptions and Limitations

用于放：

```text
未确认定义
可能 confounding
selection bias
claims lag
cost skewness
missing data
post-index leakage risk
```

规则：

1. 必须区分 “known” 和 “needs confirmation”。
2. 不要过度声称 causal conclusion。
3. 如果是观察性研究，必须有 limitation callout。

### 6.5 Footer

包含：

```text
Project name
Diagram type
Last updated date
Data privacy note
```

示例：

```text
Clinical Analytics • Study Design Visual • No PHI/PII • Draft for discussion
```


## 7. 图类型模板

### 7.1 Study Design Timeline

适用场景：

```text
HHS within 48 hours
30 day readmission
90 day cost
RAP prediction model
post-acute care analysis
```

必须展示：

1. Baseline window。
2. Index date。
3. Exposure window。
4. Outcome window。
5. Cost window。
6. 可选 censoring 或 eligibility requirement。

推荐结构：

```text
Baseline Covariates
      ↓
Index Discharge
      ↓
Exposure Window
      ↓
Outcome Window
      ↓
Cost Window
```

适合标题：

```text
Study Design Timeline
```

适合 summary cards：

```text
Cohort
Exposure Definition
Outcome Measurement
```


### 7.2 Cohort Flow

适用场景：

```text
cohort build
inclusion/exclusion
continuous enrollment
treatment/control split
analytic sample
```

必须展示：

1. Initial population。
2. Inclusion criteria。
3. Exclusion criteria。
4. Eligibility requirement。
5. Final analytic cohort。
6. Group split。

推荐结构：

```text
All inpatient discharges
→ eligible discharges
→ continuous enrollment
→ exclusions applied
→ final cohort
→ treatment/control groups
```

适合 summary cards：

```text
Starting Population
Major Exclusions
Final Unit of Analysis
```


### 7.3 Data Lineage Map

适用场景：

```text
claims + enrollment + auth + CM + provider
analytic table build
feature engineering data sources
SQL pipeline explanation
```

必须展示：

1. 原始数据源。
2. 关键 join 或 alignment。
3. 中间 analytic layer。
4. 最终 analytic table。
5. 输出用途。

推荐结构：

```text
Claims
Enrollment
Auth
Care Management
Provider
SDOH
        ↓
Analytic Feature Table
        ↓
Model or Study Output
```

适合 summary cards：

```text
Primary Keys
Join Risks
Validation Checks
```


### 7.4 Causal Framework Map

适用场景：

```text
observational study
HHS effect analysis
SNF access analysis
confounding explanation
leadership explanation of why adjustment is needed
```

必须展示：

1. Exposure。
2. Outcome。
3. Confounders。
4. Selection mechanisms。
5. Mediators or post-index variables if relevant。
6. Clear warning against causal overclaim if assumptions are not met。

推荐结构：

```text
Confounders → Exposure
Confounders → Outcome
Exposure → Outcome
Selection Context → Exposure
```

适合 summary cards：

```text
Main Confounders
Adjustment Strategy
Interpretation Limits
```


### 7.5 Analysis Pipeline

适用场景：

```text
modeling workflow
causal analysis workflow
HTE workflow
policy tree workflow
validation plan
```

必须展示：

1. Cohort construction。
2. Feature engineering。
3. Adjustment or modeling。
4. Evaluation。
5. Sensitivity analysis。
6. Business interpretation。

推荐结构：

```text
Cohort
→ Features
→ Adjustment/Model
→ Validation
→ Subgroup/HTE
→ Recommendation
```

适合 summary cards：

```text
Method
Validation
Decision Use
```


### 7.6 Result Summary Visual

适用场景：

```text
初步结果汇报
领导结论页
业务建议页
model evaluation page
```

必须展示：

1. Main finding。
2. Denominator。
3. Time window。
4. Uncertainty。
5. Limitation。
6. Recommended next action。

推荐结构：

```text
Finding
→ Evidence
→ Interpretation
→ Recommendation
```

适合 summary cards：

```text
Effect Direction
Business Impact
Caveats
```


## 8. Commands 设计

### 8.1 `visual-brief.md`

建议内容：

```md
# /visual-brief

Create a stakeholder-friendly visual brief from the current research plan, SQL logic, or analysis results.

## Required output

1. Executive takeaway
2. Recommended visual set
3. Diagram specification for each visual
4. Suggested chart types
5. Assumptions and limitations
6. Speaker-note style explanation

## Rules

- Use plain English.
- Audience is managers and directors.
- Do not overclaim causality.
- Show cohort, index date, exposure window, outcome window, and cost window when relevant.
- Prefer 3 to 5 visuals, not one giant diagram.
- If the request involves internal healthcare data, do not include PHI/PII or real row-level identifiers.
- If a visual needs more than 9 main nodes, split it.
- Prefer HTML/SVG output via the research-diagram skill when a polished diagram is requested.

## Output format

### Executive takeaway

### Recommended visual set

| Visual | Purpose | Audience | Format |
|---|---|---|---|

### Diagram specifications

For each diagram:

- Title
- Subtitle
- Diagram type
- Nodes
- Connections
- Summary cards
- Assumptions
- Speaker notes

### Chart recommendations

### Risks and limitations
```

用途：

```text
当你给 Claude 一段研究方案，让它先规划应该画哪些图。
```

不要让它一上来就生成 HTML。先让它输出视觉方案。


### 8.2 `study-diagram.md`

建议内容：

```md
# /study-diagram

Generate one polished research diagram as standalone HTML with inline SVG.

## Inputs to request if missing

- Business question
- Cohort
- Index date
- Exposure or treatment definition
- Outcome window
- Cost window if relevant
- Intended audience
- Preferred theme: light executive or dark technical

## Required process

1. Choose the best diagram type.
2. Create a concise diagram specification.
3. Use the research-diagram skill.
4. Generate a standalone `.html` file.
5. Include speaker notes and interpretation guidance.

## Rules

- One diagram should communicate one idea.
- Use no more than 9 main nodes.
- Use summary cards for details.
- Use assumptions box for uncertainty.
- Avoid PHI/PII.
- Do not include real member IDs, claim IDs, auth IDs, DOBs, names, or row-level dates.
```

用途：

```text
当你已经知道要画一张图，直接生成 HTML/SVG。
```


### 8.3 `diagram-review.md`

建议内容：

```md
# /diagram-review

Review a diagram, diagram spec, or generated HTML/SVG for clarity, accuracy, and stakeholder usefulness.

## Check

1. Is the business question visible?
2. Is the cohort clear?
3. Is the index date clear?
4. Are exposure and outcome windows clear?
5. Is the unit of analysis clear?
6. Is the visual too crowded?
7. Can a manager understand it in 30 seconds?
8. Does it overclaim causality?
9. Are assumptions and limitations visible?
10. Are PHI/PII and real identifiers absent?
11. Are colors used consistently with the visualization style guide?
12. Would the figure work in a PowerPoint screenshot?

## Output

### Verdict

### Top fixes

### Revised diagram spec

### Optional revised HTML/SVG guidance
```

用途：

```text
当你已经生成图之后，让 Claude 自查和改进。
```


## 9. Skill 设计

### 9.1 `research-diagram/SKILL.md`

建议内容：

```md
---
name: research-diagram
description: Create polished stakeholder-friendly healthcare analytics research diagrams as standalone HTML files with inline SVG. Use for study design timelines, cohort flows, data lineage maps, causal framework diagrams, analysis pipelines, and result summary visuals.
---

# Research Diagram Skill

## Goal

Generate professional research and analytics diagrams as standalone HTML files with embedded CSS and inline SVG.

The output should be suitable for:
- manager and director presentations
- analytics design reviews
- clinical analytics project documentation
- PowerPoint screenshots
- method explanation pages

## Supported diagram types

1. Study design timeline
2. Cohort flow
3. Data lineage map
4. Causal framework map
5. Analysis pipeline
6. Result summary visual

## Design principles

- Use a clean executive style by default.
- Prefer light theme unless the user requests dark technical theme.
- One diagram should communicate one main idea.
- Use semantic colors consistently.
- Use summary cards for details instead of overcrowding the main diagram.
- Use assumptions and limitations callouts.
- Avoid PHI/PII and real row-level identifiers.
- Do not overclaim causality for observational studies.

## Required inputs

Before generating the final diagram, infer or ask for:

- Diagram type
- Business question
- Audience
- Cohort
- Index date
- Exposure or treatment definition
- Outcome window
- Cost window if relevant
- Main assumptions
- Main limitations

If some inputs are missing, use placeholders and mark them clearly as `Needs confirmation`.

## Output requirements

Always produce one standalone `.html` file with:

- embedded CSS
- inline SVG
- no external images
- no JavaScript required
- responsive layout
- title and subtitle
- main diagram
- three summary cards
- assumptions and limitations section
- footer metadata

## Layout rules

- Maximum 9 main nodes.
- Minimum 32 px gap between nodes.
- Arrows must not cross text.
- Legend must not overlap with nodes.
- Text inside nodes must fit within the node.
- Use callouts for uncertainty instead of long labels.
- If a diagram becomes crowded, split it into multiple diagrams.

## Semantic colors

- Data Source: blue
- Cohort: cyan
- Exposure: green
- Outcome: purple
- Method: yellow
- Confounder: orange
- Risk or Limitation: red
- External Context: slate

## Required final response

After creating the HTML file, summarize:

1. What visual was created
2. What assumptions were used
3. What the user should review before sharing
```


### 9.2 `base_template.html` 设计要求

不用一开始写非常复杂。模板结构如下：

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{TITLE}}</title>
  <style>
    :root {
      --bg: #F8FAFC;
      --card: #FFFFFF;
      --text: #0F172A;
      --muted: #475569;
      --border: #CBD5E1;
      --data: #2563EB;
      --cohort: #0891B2;
      --exposure: #16A34A;
      --outcome: #7C3AED;
      --method: #CA8A04;
      --confounder: #EA580C;
      --risk: #DC2626;
      --external: #64748B;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Aptos, Segoe UI, Arial, sans-serif;
    }

    .page {
      max-width: 1180px;
      margin: 32px auto;
      padding: 0 24px;
    }

    .header {
      margin-bottom: 20px;
    }

    .eyebrow {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }

    h1 {
      font-size: 28px;
      line-height: 1.2;
      margin: 0 0 8px 0;
    }

    .subtitle {
      color: var(--muted);
      font-size: 15px;
      max-width: 900px;
    }

    .diagram-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    svg {
      width: 100%;
      height: auto;
      display: block;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-top: 18px;
    }

    .summary-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
    }

    .summary-card h3 {
      font-size: 14px;
      margin: 0 0 8px 0;
    }

    .summary-card ul {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .limitations {
      margin-top: 16px;
      padding: 14px 16px;
      border-left: 4px solid var(--risk);
      background: #FEF2F2;
      border-radius: 12px;
      color: #7F1D1D;
      font-size: 13px;
      line-height: 1.5;
    }

    .footer {
      color: var(--muted);
      font-size: 12px;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="header">
      <div class="eyebrow">{{DIAGRAM_TYPE}}</div>
      <h1>{{TITLE}}</h1>
      <div class="subtitle">{{SUBTITLE}}</div>
    </section>

    <section class="diagram-card">
      {{INLINE_SVG}}
    </section>

    <section class="summary-grid">
      {{SUMMARY_CARDS}}
    </section>

    <section class="limitations">
      {{ASSUMPTIONS_AND_LIMITATIONS}}
    </section>

    <section class="footer">
      {{FOOTER}}
    </section>
  </main>
</body>
</html>
```


## 10. Reference 文件设计

### 10.1 `visual_style_guide.md`

建议内容：

```md
# Visualization Style Guide

## Audience

Primary audience:
- managers
- directors
- clinical analytics stakeholders
- care management stakeholders

Assume the audience understands healthcare operations but may not know SQL, causal inference, or machine learning details.

## Core principles

- One visual should make one point.
- Put the business question in the title or subtitle.
- Use plain English labels.
- Avoid internal table names unless the diagram is for technical review.
- Avoid unexplained abbreviations.
- Show time anchors explicitly.
- Separate observed facts from assumptions.
- Separate association from causal interpretation.
- Use summary cards for detail.
- Keep the main diagram visually sparse.

## Default theme

Use Light Executive Theme unless the user requests Dark Technical Theme.

## Semantic colors

- Data Source: blue
- Cohort: cyan
- Exposure: green
- Outcome: purple
- Method: yellow
- Confounder: orange
- Risk or Limitation: red
- External Context: slate

## Layout

- Maximum 9 main nodes.
- Use left-to-right flow for process diagrams.
- Use top-to-bottom flow for funnel or cohort attrition diagrams.
- Keep legend outside the main flow.
- Do not let arrows cross labels.
- Use callouts for assumptions.
```

### 10.2 `research_visual_templates.md`

建议内容：

```md
# Research Visual Templates

## Study Design Timeline

Use for:
- baseline/index/outcome window
- HHS within 48h
- readmission outcome
- cost follow-up

Required elements:
- baseline window
- index date
- exposure window
- outcome window
- cost window if relevant
- eligibility requirement

## Cohort Flow

Use for:
- inclusion/exclusion
- cohort attrition
- treatment/control split

Required elements:
- starting population
- inclusion criteria
- exclusion criteria
- final cohort
- unit of analysis
- group split

## Data Lineage Map

Use for:
- data source explanation
- SQL pipeline
- analytic table build

Required elements:
- source tables
- key joins
- final analytic table
- validation checks

## Causal Framework Map

Use for:
- explaining confounding
- observational study design
- HTE rationale

Required elements:
- exposure
- outcome
- confounders
- selection mechanism
- adjustment strategy
- limitations

## Analysis Pipeline

Use for:
- modeling workflow
- causal workflow
- validation plan

Required elements:
- cohort build
- feature engineering
- adjustment/modeling
- validation
- interpretation
```

### 10.3 `chart_rules.md`

建议内容：

```md
# Chart Rules for Healthcare Analytics

## General rules

- Every chart needs a clear title and subtitle.
- State denominator and time window.
- Show N where relevant.
- Avoid 3D charts.
- Avoid dual-axis charts unless explicitly justified.
- Prefer rates with denominators over raw counts when comparing groups.
- For cost, show distribution, not only mean.
- For skewed cost, include median/IQR or sensitivity view.
- For causal analyses, separate unadjusted and adjusted results.

## Common chart choices

- Readmission rate by group: bar chart with confidence interval
- Monthly trend: line chart
- Cost distribution: box plot, violin plot, or percentile table
- Balance before/after weighting: love plot
- CATE distribution: histogram or density plot
- Subgroup effect: forest plot
- Cohort attrition: flow diagram
- Model calibration: calibration plot
- Model discrimination: ROC or PR curve
```

### 10.4 `diagram_antipatterns.md`

建议内容：

```md
# Diagram Antipatterns

## Antipattern 1: One giant diagram

Problem:
Combines cohort flow, timeline, data lineage, causal assumptions, and results in one figure.

Better:
Split into separate visuals.

## Antipattern 2: No time anchor

Problem:
Shows exposure and outcome but not index date or windows.

Better:
Always show baseline, index, and outcome window for longitudinal analyses.

## Antipattern 3: Overclaiming causality

Problem:
Uses labels like "HHS reduces readmission" before causal assumptions are justified.

Better:
Use "estimated association" or "adjusted comparison" unless causal identification is credible.

## Antipattern 4: Too many unexplained acronyms

Problem:
Uses HHS, SNF, RAP, IP, CM, PA without explanation.

Better:
Expand or define terms in labels or summary cards.

## Antipattern 5: Technical table names in executive visuals

Problem:
Uses raw table names in leadership diagrams.

Better:
Use business labels, and reserve table names for technical lineage diagrams.
```


## 11. 与 SQL harness 的配合

你的 SQL harness 负责：

```text
table_catalog.md
join_registry.md
business_definitions.md
known_pitfalls.md
antipatterns.md
```

视觉 harness 负责：

```text
visual_style_guide.md
research_visual_templates.md
chart_rules.md
diagram_antipatterns.md
```

二者配合方式：

```text
SQL harness 保证分析逻辑正确
visual harness 保证表达清楚、美观、适合汇报
```

当图涉及真实数据逻辑时，视觉命令应先读取 SQL/domain references，防止图把业务口径画错。

示例规则：

```text
如果图涉及 cohort、join、cost、readmission、HHS/SNF 定义：
先参考 company-data references
再生成视觉图
```


## 12. 推荐工作流

### 场景 A：从研究方案生成视觉包

```text
/visual-brief
输入：研究方案文字
输出：建议 3 到 5 张图及每张图的 diagram spec
```

然后：

```text
/study-diagram
输入：其中一张 diagram spec
输出：HTML/SVG 图
```

最后：

```text
/diagram-review
输入：HTML 或 diagram spec
输出：修改建议
```


### 场景 B：从 SQL 生成数据 lineage 图

步骤：

1. Claude 先用 internal SQL rule 审核 SQL。
2. 提取 source tables、joins、final output grain。
3. 生成 data lineage diagram spec。
4. 用 research-diagram skill 生成 HTML/SVG。
5. 用 diagram-review 检查是否过度技术化。


### 场景 C：从结果表生成领导汇报图

步骤：

1. 明确业务问题。
2. 明确 denominator、time window、comparison group。
3. 选择图表类型。
4. 生成 result summary visual。
5. 加 assumptions and limitations。
6. 输出为 HTML/SVG 或后续转 PPT。


## 13. 第一阶段落地计划

### 第 1 天

建文件：

```text
.claude/commands/visual-brief.md
.claude/commands/diagram-review.md
.claude/references/visualization/visual_style_guide.md
.claude/references/visualization/research_visual_templates.md
.claude/references/visualization/chart_rules.md
.claude/skills/research-diagram/SKILL.md
.claude/skills/research-diagram/assets/base_template.html
```

### 第 2 到 3 天

用 2 个真实项目测试：

1. HHS within 48h after discharge study design。
2. RAP 30 day readmission model workflow。

每个项目生成：

```text
study design timeline
cohort flow
analysis pipeline
```

### 第 1 周结束

补充：

```text
diagram_antipatterns.md
cohort_flow_template.html
study_design_template.html
```

### 第 2 周

把最常用的 3 个模板稳定下来：

```text
study_design_template.html
cohort_flow_template.html
data_lineage_template.html
```

### 第 3 到 4 周

再考虑：

```text
causal_framework_template.html
analysis_pipeline_template.html
result_summary_template.html
PowerPoint export workflow
```


## 14. 成功标准

第一阶段成功标准不是“自动生成完美 PPT”。

成功标准是：

1. Claude 能稳定把研究方案拆成 3 到 5 张视觉图。
2. 每张图能清楚展示 cohort、index date、window 和主要 assumption。
3. 图比 Mermaid 更适合放进汇报材料。
4. 领导或 manager 能在 30 秒内理解每张图的核心意思。
5. 图不泄露 PHI/PII。
6. 图不会过度声称 causal conclusion。
7. 图的风格一致，不像每次临时画出来的。


## 15. 最终推荐

我建议你不要只安装一个现成 architecture diagram skill，而是借鉴它的实现方式，做一个适合你工作的 healthcare analytics research diagram generator。

最终形态：

```text
SQL harness 保证分析逻辑正确
Research diagram harness 保证表达清楚、美观、适合汇报
Commands 负责触发和 review
Skill 负责 HTML/SVG 渲染
Templates 负责稳定版式
References 负责视觉和统计表达规则
```

最小可行版本：

```text
visual-brief.md
diagram-review.md
visual_style_guide.md
research_visual_templates.md
chart_rules.md
research-diagram/SKILL.md
base_template.html
```

这个版本已经足够帮助你把 HHS/SNF/readmission/cost/RAP 这类复杂研究，用更形象、更漂亮、更适合领导理解的方式表达出来。
