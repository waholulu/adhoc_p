我复核完后，最终建议是：只做一个极简领域 skill，不再组合多个社区 skill 常驻。

官方文档里有几个关键原则直接支持这个方向：Skill 只需要一个带 YAML frontmatter 的 SKILL.md，必填 name 和 description；Claude 会先加载 skill metadata，只有相关时才读完整 SKILL.md，但一旦读入，内容仍会占用上下文，所以官方明确建议 Skill 要 concise。官方还建议 description 要具体说明“做什么、何时用”，并且多个 focused skills 比一个大而全 skill 更好。 ￼

我参考的几个 skill 里，真正值得吸收的是：management-consulting 的 decision first、MECE、evidence labels、risks、kill conditions、next actions；mckinsey-critic 的 top 3 fixes 和 claim-title 审稿；data-storytelling 的 chart-level takeaway；ppt-creator 的 assertion-style headings。最终文档没有照搬它们的大框架，而是压缩成一个适合你 healthcare analytics 场景的工作流。 ￼

下面就是我确认后的最终版 SKILL.md。

---
name: healthcare-analytics-storyline
description: Converts healthcare analytics, claims data, model evaluation, cohort comparison, readmission risk, post-acute care, and observational analysis results into manager/director-facing storylines. Use when turning analysis outputs into an executive narrative, memo outline, slide outline, or report structure. Do not use for general strategy consulting, visual design, PPTX rendering, HTML styling, or file generation.
---
# Healthcare Analytics Storyline
Use this skill to turn analysis outputs into a clear, decision-useful, scientifically honest stakeholder narrative.
Default behavior: create a Narrative Blueprint, not slides, HTML, or files.
## Core workflow
1. Define the decision
2. State the one-sentence answer
3. Build the evidence map
4. Separate facts from inferences and assumptions
5. Create the storyline
6. Add operational implications
7. Preserve important caveats
8. Review the top 3 fixes before stakeholder delivery
## Required output
### 1. Decision or business question
State the decision this analysis informs.
Include:
- Audience
- Decision-maker
- Decision to support
- Success metric, if known
If information is missing, state provisional assumptions and continue.
### 2. One-sentence answer
Give the main answer in one sentence.
The answer should be clear enough that a manager or director can understand it in 30 seconds.
### 3. Evidence map
Use this format:
| Claim | Evidence | Label | Confidence | Caveat |
|---|---|---|---|---|
| Claim or finding | Metric, chart, table, model output, or source | Fact / Inference / Assumption / Estimate | High / Medium / Low | Limitation or condition |
Label every major claim:
- Fact: directly observed in the data
- Inference: derived from facts
- Assumption: judgment used to fill a gap
- Estimate: calculated using data plus assumptions
Never present assumptions or inferences as facts.
### 4. Causal caution
For observational claims data:
- Do not imply causality unless there is an experiment or quasi-experimental design.
- Causal claims must be labeled as Inference unless the identification strategy is stated.
- State the main identification assumptions when discussing causal effects.
- Preserve uncertainty when it affects operational decisions, trust, model adoption, or interpretation.
Bad:
"Home health within 48 hours reduces readmissions."
Better:
"Members receiving home health within 48 hours had lower observed readmission rates; this is an association unless selection bias is addressed."
### 5. Storyline
Use this structure:
1. Main answer
2. Supporting point 1
3. Supporting point 2
4. Supporting point 3
5. Operational implication
6. Caveats
7. Next actions
Use claim-based headings, not topic headings.
Bad:
"Model Performance"
Better:
"The top risk deciles concentrate enough readmissions to support targeted outreach."
### 6. Chart and table discipline
For every chart, table, or model output, include:
- What to notice
- Why it matters
- What decision it supports
- Whether it shows fact, inference, assumption, or estimate
Do not merely describe the visual.
Bad:
"This chart shows readmission rate by risk decile."
Better:
"The top two risk deciles capture a disproportionate share of readmissions, supporting a focused outreach list when care management capacity is limited."
### 7. Operational implications
Connect recommendations to workflow.
For care management or clinical operations, specify:
- Who uses the output
- When they use it
- What action changes
- How many members or cases are affected
- What capacity constraint matters
- What feedback loop is needed
- What success metric should be tracked
Recommendations must be operationally usable, not just analytically interesting.
### 8. Risks and caveats
Include caveats that affect:
- Causal interpretation
- Operational safety
- Model adoption
- Stakeholder trust
- Fairness or subgroup performance
- Data quality
- Claims lag
- Workflow capacity
Do not remove important caveats just to make the recommendation sound cleaner.
### 9. Kill conditions
List what evidence would weaken or reverse the recommendation.
Examples:
- The finding disappears after adjustment for baseline acuity.
- Performance drops materially in a key subgroup.
- Care management capacity cannot support the proposed outreach volume.
- The operational team cannot act on the score within the needed time window.
- Prospective monitoring shows no lift versus current prioritization.
### 10. Top 3 fixes
Before finalizing, review the output and list the top 3 fixes.
Check for:
1. Unclear main answer
2. Unsupported claims
3. Causal overstatement
4. Missing operational implication
5. Topic titles instead of claim titles
6. Chart captions that describe but do not interpret
7. Caveats that are missing or overemphasized
8. Recommendations that are not actionable
## Output format
Use this structure unless the user asks for a different format:
```markdown
# Narrative Blueprint
## 1. Decision Question
...
## 2. One-Sentence Answer
...
## 3. Audience and Decision Context
...
## 4. Evidence Map
| Claim | Evidence | Label | Confidence | Caveat |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
## 5. Storyline
### Main answer
...
### Supporting point 1
...
### Supporting point 2
...
### Supporting point 3
...
## 6. Operational Implications
...
## 7. Risks, Caveats, and Kill Conditions
...
## 8. Recommended Next Actions
...
## 9. Top 3 Fixes Before Stakeholder Delivery
1. ...
2. ...
3. ...

Style rules

* Start with the so what.
* Be concise.
* Use simple business language.
* Avoid generic consulting frameworks unless they directly improve the healthcare analytics decision.
* Do not data dump.
* Do not show methodology first unless trust depends on it.
* Use claim-based headings.
* Preserve decision-relevant uncertainty.
* Make the narrative useful for managers, directors, care management teams, and healthcare analytics stakeholders.

## 我建议的文件结构
```text
.claude/
  skills/
    healthcare-analytics-storyline/
      SKILL.md

或者如果你想全局使用：

~/.claude/
  skills/
    healthcare-analytics-storyline/
      SKILL.md

不建议再加的内容

我不建议在这个 skill 里再塞：

MECE 的长解释
42 个咨询框架
完整 McKinsey critic rubric
PPT 版式规则
HTML 设计规则
python-pptx 脚本
参考资料文件夹

原因很简单：你的目标是让 Claude 先把分析讲清楚，不是让它在一个 skill 里完成“咨询分析 + 质检 + PPT 设计 + 文件生成”。Mck-ppt-design 这种需要 Bash、python-pptx、lxml 的 production skill，仍然只适合最后在 sandbox 单独用。 ￼

最终使用方式

以后你可以这样触发：

Use healthcare-analytics-storyline.
先不要做 PPT。请把这些代码分析结果整理成 Narrative Blueprint，面向 manager/director 和 care management stakeholders。

需要转 PPT outline 时再说：

Based on the approved Narrative Blueprint, create a slide outline. Do not change the logic, evidence labels, or caveats.

这个版本是我认为最简洁、最稳、不容易 context 膨胀的最终 skill 文档。