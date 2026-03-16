# Modern Data Research Automation Workflow: A Spec-Driven & RPVI Architecture

This central document outlines an AI-assisted research and complex data analysis workflow that integrates **Spec-driven development** with the **RPVI (Read-Plan-Verify-Iterate)** framework. By deeply parameterizing underlying data structures and high-dimensional business logic, this workflow leverages Large Language Models (LLMs) to achieve end-to-end automation—from logic definition and code generation to insight delivery.

---

## 1. Spec-Driven Development Explained

In vibe coding, spec-driven development means shifting from “just prompt the AI and iterate” to first writing an explicit spec—covering the feature’s behavior, constraints, interfaces, edge cases, and expected outcomes—so the model generates code against a clearer contract instead of guessing from a loose prompt; this is why recent tooling and writeups frame it as a more predictable, maintainable alternative for projects that are bigger than quick prototypes, with examples like GitHub’s Spec Kit emphasizing “product scenarios and predictable outcomes instead of vibe coding every piece from scratch,” and other frameworks describing a workflow of requirements/design/tasks/execution rather than pure prompt-and-patch iteration.
In traditional analytics workflows, business logic is often scattered across analysts' memories, fragmented emails, or buried within thousands of lines of SQL/Python code ("Code is Truth"). In the **Spec-driven** paradigm, the **Specification (Spec)** serves as the absolute and only **Single Source of Truth**.

We do not ask the AI to write code directly; instead, we instruct the AI to "read and strictly adhere to" predefined specifications.



**A. Anatomy of the Specification (The Spec)**
A high-quality Spec (typically a `.md` or `.yaml` file) must encompass the following three dimensions to entirely eliminate ambiguity in the business context:
1. **Data Ontology & Schema**:
   * Clearly define table structures, field definitions, and physical storage locations.
   * Strictly define **Join Keys** (e.g., explicitly mandating that cross-table mapping must use `member_id` combined with a specific `date_window`, rather than allowing the AI to improvise).
2. **Business Logics & Definitions**:
   * Make complex computational logic explicit. For instance, explicitly defining the inclusion/exclusion criteria for a "valid sample", or providing the exact mathematical formulas and filtering conditions for metrics like the "30-day readmission rate" or "anomalous claims" (e.g., explicitly excluding records where the status is `reversed`).
3. **Engineering Constraints**:
   * Enforce code output standards (e.g., requiring the use of CTEs instead of nested subqueries, or mandating comprehensive exception handling and logging in Python scripts).


---

## 2. Workflow Execution: The Four Stages of Cursor Agentic Implementation

This workflow engineers the theoretical framework above into practical steps using advanced IDEs like Cursor, perfectly mapping to the RPVI framework:

### Stage 1: Parameterizing Domain Knowledge (Read & Define: The Knowledge Spec)
* **Action**: Upon project initialization, establish a dedicated `docs/` directory. Distill all `Key Tables`, `Join Conditions`, core hypotheses, and `Business Logics` into independent, structured `.md` files.
* **Value**: This is the foundation of the entire automation process. It serves not only as the blueprint for AI code generation but also as the standard alignment document for human researchers.

### Stage 2: Agentic Execution & Engineering (Plan & Implement)
* **Action**: Based on the project brief and the predefined Spec files, instruct the Cursor Agent to plan the architecture and generate executable code:
    * **BigQuery / SQL Scripts**: Generate complex data extraction, cleaning, and feature engineering scripts based on the Spec, ensuring join logic is executed flawlessly.
    * **Python Automation Scripts**: Generate code for data processing, statistical testing, and report compilation. Specifically, mandate the generation of a **QC Stats (Quality Control Statistics)** module to output intermediate metrics (e.g., row counts, missingness rates, core field distributions) for bidirectional validation against external benchmarks.
    * **Methodology Doc**: Concurrently require the Agent to reverse-engineer a methodology document from the generated code, ensuring the "black box" process becomes transparent and auditable.

### Stage 3: Cognitive Enhancement & Initial Validation (Verify - Step 1: Internal QC)
* **Action**: Utilize Cursor's chat system or directly call LLM APIs to feed the Python-generated analytical results and QC reports back to the AI. Ask it to summarize key findings and perform Anomaly Detection based on the initial project brief.
* **Value**: Leverages the LLM's powerful information compression and extraction capabilities to trace tedious structured data and statistical features back to the business context, verifying that the outputs align with the original research intent.

### Stage 4: Multi-Model Consensus Validation & Iteration (Verify - Step 2 & Iterate: Cross-model Consensus)
* **Action**: Introduce a "Multi-Expert Consultation" mechanism. Pass the key data findings and analytical conclusions (anonymized/desensitized) to top-tier LLMs of different architectures (e.g., switching from the code-heavy reasoning of Claude 3.5 Sonnet to the divergent logic of GPT-4o or Gemini 1.5 Pro) for secondary or tertiary independent review.



* **Value**: This represents the highest level of **Verification**. Different models have varying training distributions and reasoning biases. Cross-model Consensus effectively hedges against the logical blind spots of any single model. Consistent conclusions across models increase confidence; discrepancies accurately pinpoint issues, driving researchers back to Stage 1 to update the Spec, thus completing the **Iteration** loop.

---

## 3. Workflow Value Summary & Comparison

| Core Dimension | Traditional Manual Approach (Code-centric) | Modern AI-Enhanced Approach (Spec-driven) |
| :--- | :--- | :--- |
| **Logical Consistency & Maintenance** | Heavily relies on analyst memory. Personnel changes lead to lost context; code maintenance costs are extremely high. | **Spec as Document, Spec as Code.** Strict constraints yield near-zero hallucinations. Business logic updates only require document changes. |
| **Execution & Refactoring Efficiency** | Hand-coding complex SQL and reports. Requirement changes demand line-by-line edits, taking days to weeks. | Automated script orchestration, **"One-Click Refactoring."** Changes in business logic just require a Spec update, and the Agent regenerates the full codebase in minutes. |
| **Reliability & Quality Control** | Relies on manual spot-checks. Narrow coverage, prone to "Silent Errors." | **Comprehensive automated QC + Multi-model cross-validation**, ensuring machine-level rigor. |
| **Asset Accumulation Value** | Typically only yields the final PPT or static dashboard; process assets are lost. | **Full-funnel digitization**: From business specs and underlying code to QC logic and final reports, creating a complete, reusable digital asset package. |
