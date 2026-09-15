"""
Builds the grounded prompt sent to the LLM.

The prompt injects three things the model must not invent:

  1. the mathematical output of the Random Forest (probability + band),
  2. the WHO/IDF interpretation of every submitted value, and
  3. the WHO guidance the closing paragraph must draw on.

Grounding the model this way keeps the narrative tethered to the clinical facts
in `who_idf_thresholds.json` instead of the model's own recollection.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.knowledgebase.clinical_reference import (
    ParameterFinding,
    get_reference,
    rank_by_influence,
)


SYSTEM_PROMPT = """You are a clinical communication assistant embedded in a diabetes \
screening tool used at primary health centres in Nigeria. You translate the output of a \
Random Forest risk model into plain, calm language a patient or a community health worker \
can act on.

Rules you must follow without exception:

1. NEVER diagnose. This tool screens; it does not diagnose. Say "suggests", "is associated \
with", "warrants testing" - never "you have diabetes".
2. Use ONLY the clinical facts supplied in the CLINICAL REFERENCE section of the user \
message. Do not introduce thresholds, statistics, or guidance from memory.
3. Do not restate the probability as a certainty and do not invent a second number. The \
probability given to you is the only figure you may cite.
4. Never recommend a specific medication, dose, or treatment. You may recommend tests, \
lifestyle change, and seeing a clinician.
5. Write for a reader with no medical training. Expand any abbreviation on first use. \
Short sentences. No bullet symbols, no markdown, no headings.
6. Be direct but not alarming. A high result should read as urgent and actionable, not \
frightening.
7. British English spelling. Do not address the reader by name or invent personal details.

Output format - return exactly three paragraphs separated by a single blank line, and \
nothing else (no preamble, no closing sign-off, no headings):

Paragraph 1 (2-3 sentences): what the risk score means for this person, referring to the \
band name.
Paragraph 2 (3-4 sentences): which of their specific measurements drove the result, quoting \
the value and the WHO/IDF band it falls in.
Paragraph 3 (2-3 sentences): the concrete next step, drawn from the recommended action \
supplied to you."""


#: Guideline blocks to inject, at most. Grounding needs the parameters the
#: narrative will actually name; beyond that the extra text only dilutes the
#: prompt and burns tokens against the provider's per-minute limit.
MAX_GUIDELINES = 2

#: General WHO guidance the closing paragraph draws on. The other entries in
#: the knowledge base are shown on the pages, not needed by the writer.
RELEVANT_GUIDANCE_IDS = ("lifestyle", "confirmation")


def _format_findings(findings: List[ParameterFinding]) -> str:
    """
    One line per measurement.

    Abnormal values carry their explanatory note, because the narrative has to
    describe them. Normal values are listed by name and band only - the model
    needs to know they were in range, not why.
    """
    lines = []
    for finding in findings:
        value = f"{finding.value:g}"
        unit = "" if finding.unit in ("count", "score") else f" {finding.unit}"
        if finding.is_flagged:
            marker = "OUTSIDE RANGE" if finding.severity == "critical" else "BORDERLINE"
            lines.append(
                f"- {finding.display_name}: {value}{unit} "
                f"[{marker}] -> {finding.band_label}. {finding.note}"
            )
        else:
            lines.append(
                f"- {finding.display_name}: {value}{unit} [normal] -> {finding.band_label}"
            )
    return "\n".join(lines)


def _format_guidelines(findings: List[ParameterFinding]) -> str:
    """
    Inject guideline text only for the parameters the narrative will discuss.

    `findings` arrives ranked by influence, so taking the first few flagged
    entries gives the model the facts behind exactly the measurements it is
    about to name.
    """
    flagged = [f for f in findings if f.is_flagged]
    selected = (flagged or findings)[:MAX_GUIDELINES]

    seen: set[str] = set()
    lines = []
    for finding in selected:
        if finding.guideline in seen:
            continue
        seen.add(finding.guideline)
        lines.append(f"- On {finding.display_name}: {finding.guideline}")
    return "\n".join(lines)


def build_narrative_prompt(
    probability: float,
    findings: List[ParameterFinding],
    top_drivers: List[Dict[str, Any]] | None = None,
) -> str:
    """
    Assemble the user-turn prompt.

    Args:
        probability: Random Forest probability of the positive class, 0.0-1.0.
        findings: WHO/IDF interpretation of each submitted value.
        top_drivers: Optional feature-importance rows from the model, so the
            narrative can prioritise the parameters the model actually weighted.
    """
    reference = get_reference()
    band = reference.band_for_probability(probability)
    percent = round(probability * 100, 1)

    # Present the measurements in the order the model actually weights them, so
    # the guideline block below covers exactly the parameters the narrative will
    # lead with. Without this the prompt can name Glucose as the top driver
    # while supplying the WHO facts for something else.
    findings = rank_by_influence(findings, top_drivers)

    drivers_block = ""
    if top_drivers:
        ranked = ", ".join(
            f"{d['display_name']} ({d['importance'] * 100:.0f}% of model weight)"
            for d in top_drivers[:3]
        )
        drivers_block = (
            "\nMODEL WEIGHTING\n"
            f"Across the training cohort the model relies most on: {ranked}.\n"
        )

    guidance = "\n".join(
        f"- {item['title']}: {item['body']}"
        for item in reference.general_guidance
        if item["id"] in RELEVANT_GUIDANCE_IDS
    )

    return f"""MODEL OUTPUT
Probability of diabetes: {percent}%
Risk band: {band['label']}
What this band means: {band['summary']}
Recommended action for this band: {band['action']}
{drivers_block}
PATIENT MEASUREMENTS, INTERPRETED AGAINST WHO/IDF THRESHOLDS
{_format_findings(findings)}

CLINICAL REFERENCE - these are the only clinical facts you may use
{_format_guidelines(findings)}

GENERAL WHO GUIDANCE
{guidance}

Write the three-paragraph explanation now, following every rule in your instructions."""
