我又想了一遍：上一版适合“数据分析汇报”，但对“因果分析为基础的汇报”还不够硬。

它已经有 causal caution，但还是偏“提醒不要乱说 causality”。如果你真的要用它服务于 HHS 是否降低 readmission、SNF 替代方案是否节省成本、早期干预是否有效这类问题，它还需要多一层：

不只是提醒 causal claim 要谨慎，而是强制 Claude 先定义 causal question、estimand、comparison、time window、identification assumptions。

否则 Claude 还是容易写出“看起来很稳的商业叙事”，但因果识别部分是松的。

我的判断

对一般数据分析汇报

适合度：85%

它已经能解决：

1. 不 data dump
2. 先讲 so what
3. 每张图有 takeaway
4. claim 要有 evidence
5. recommendation 要接 workflow
6. caveat 不乱删

这个对模型评估、risk decile、subgroup analysis、cohort profile、trend analysis 都够用。

对因果分析汇报

适合度：70%

主要不足是少了这几个硬约束：

1. 没强制定义 estimand
2. 没强制 treatment/control/index date/outcome/follow-up
3. 没强制说明 identification strategy
4. 没强制检查 confounding、selection bias、immortal time bias
5. 没强制展示 balance/overlap/sensitivity
6. 没强制区分 descriptive finding vs causal estimate vs policy recommendation

对于 causal analysis，这些比“故事讲清楚”更基础。

我建议的修改方向

不要把整个 skill 改成很重的 causal inference 教科书。
最好的办法是加一个按需触发的 Causal Evidence Mode。

也就是：

普通数据分析 → 走 Narrative Blueprint
涉及 effect / impact / reduce / save / intervention / treatment / causal question → 自动加 Causal Evidence Box

这样既不会让 context 太膨胀，也能防止关键场景跑偏。

建议加入的核心模块

把下面这一段加进 SKILL.md，放在 Causal caution 后面。

### Causal Evidence Mode
Use this mode when the analysis asks whether an intervention, service, program, outreach, model action, or exposure affects an outcome.
Trigger examples:
- Does home health within 48 hours reduce readmissions?
- Does outreach lower cost or utilization?
- Does SNF versus home health change outcomes?
- What is the impact of an intervention?
- Should we expand this program based on observed outcomes?
When Causal Evidence Mode is triggered, include a Causal Evidence Box.
#### Causal Evidence Box
State:
1. Causal question
   - What effect are we trying to estimate?
2. Estimand
   - Target population
   - Treatment or exposure
   - Comparison group
   - Outcome
   - Time horizon
3. Time alignment
   - Index date
   - Treatment window
   - Baseline covariate window
   - Follow-up window
   - Exclusion rules
4. Identification strategy
   - Experiment, quasi-experiment, matching, weighting, regression adjustment, difference-in-differences, instrumental variable, regression discontinuity, or descriptive comparison
   - If descriptive only, say so clearly
5. Key assumptions
   - Exchangeability or no unmeasured confounding
   - Positivity or overlap
   - Consistency
   - Correct time ordering
   - Stable outcome measurement
6. Main threats to validity
   - Confounding by acuity or care need
   - Selection bias
   - Immortal time bias
   - Regression to the mean
   - Claims lag or coding variation
   - Missing data
   - Differential follow-up
   - Capacity or access constraints
7. Diagnostics or robustness checks
   - Baseline balance
   - Overlap or propensity score distribution
   - Sensitivity analysis
   - Subgroup checks
   - Negative control, if available
   - Alternative outcome or follow-up windows
   - Pre-trend check for difference-in-differences
8. Interpretation level
   Label the result as one of:
   - Descriptive association
   - Adjusted association
   - Causal estimate with assumptions
   - Experimental estimate
   - Operational recommendation
9. Safe wording
   Use language that matches the evidence strength.

我还会改几个现有句子

原句

Causal claims from observational claims data are Inference, not Fact, unless backed by an experiment or quasi-experimental design with stated identification assumptions.

建议改成

Causal claims from observational claims data are not Facts. They are causal estimates or causal inferences only when the estimand, comparison group, time alignment, identification strategy, and key assumptions are stated.

这个更准确。因为即使用 quasi-experimental design，也不是 Fact，而是基于识别假设的 causal estimate。

还要加一个“语言分级”规则

这个对汇报特别重要。Claude 很容易把 association 写成 effect。

建议加入：

### Evidence-strength wording
Match wording to evidence strength:
- Descriptive association:
  "Members who received X had lower observed Y."
- Adjusted association:
  "After adjustment, X was associated with lower Y."
- Causal estimate with assumptions:
  "Under the stated identification assumptions, X is estimated to reduce Y."
- Experimental evidence:
  "Randomized evidence shows X reduced Y."
Avoid causal verbs such as "reduce," "prevent," "save," "drive," or "cause" unless the causal evidence level supports them.

这条非常适合你。
因为你汇报里最容易出问题的就是标题写成：

Early home health reduces readmissions

但更稳的标题应该是：

Early home health is associated with lower readmission risk, but selection bias must be addressed before treating this as causal

或者如果你有严谨设计：

After weighting and balance checks, early home health is estimated to lower readmissions under no-unmeasured-confounding assumptions

我建议的最终结构

你的 skill 结构可以变成这样：

1. Decision or business question
2. One-sentence answer
3. Evidence map
4. Causal Evidence Mode, only if applicable
5. Storyline
6. Chart and table discipline
7. Operational implications
8. Risks and caveats
9. Kill conditions
10. Top 3 fixes

也就是只多加一个 causal mode，不把整个 skill 变重。

最终判断

我会这样说：

上一版适合数据分析汇报，但对因果分析汇报还需要补强。
补强重点不是加更多咨询框架，而是加 Causal Evidence Box 和 evidence-strength wording。

加完以后，这个 skill 才真正适合你的核心工作：

claims data 分析
cohort comparison
readmission risk model
home health/SNF/post-acute care opportunity
care management intervention evaluation
observational causal inference
manager/director-facing recommendation

我的最终推荐

采用这个版本：

healthcare-analytics-storyline
= narrative blueprint
+ chart-level takeaway
+ operational implication
+ causal evidence mode
+ evidence-strength wording
+ top 3 fixes

不要再加完整 causal inference 教科书，也不要装一堆额外 skill。
只需要把上面的 Causal Evidence Mode 和 Evidence-strength wording 加进去，整个 skill 就更适合“数据分析 + 因果分析为基础的汇报”。