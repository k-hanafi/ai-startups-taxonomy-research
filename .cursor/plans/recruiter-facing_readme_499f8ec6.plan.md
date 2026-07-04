---
name: Recruiter-facing README
overview: Replace the current technical README with a concise, timeless, recruiter-facing README that communicates the research goals, the LLM directness concept, and intellectual rigor -- without implementation details that will go stale.
todos:
  - id: write-readme
    content: Write the new README.md replacing the current 200-line technical README with the 7-section recruiter-facing version
    status: completed
isProject: false
---

# Recruiter-Facing README for LLM Directness Experiment

## Design Principles

- **Timeless**: No specific model names (e.g., "gpt-5.4-nano"), no current results, no CLI usage, no quickstart instructions. Describe the *approach* and *why*, not the *current state*.
- **Recruiter-readable**: Assume intelligent non-technical reader. Lead with the research question, not the code.
- **Concise**: Target ~80-100 lines. Current README is 200 lines of operational detail; this should be roughly half that length and all substance.
- **Shows depth**: The experimental design and statistical methodology sections demonstrate research rigor without requiring domain expertise to appreciate.

## Proposed Sections

### 1. Title + Tagline (2-3 lines)

Keep the existing title "LLM Directness Experiment." Follow with a one-sentence tagline that frames the research question clearly:

> *When an LLM classifies a company, is its verdict based on the input it was shown -- or on facts it memorized during pretraining?*

### 2. Motivation (~2 short paragraphs)

Establish the author's role and the pipeline they built. Key beats:
- As part of research work at UBC on [Bena, Bian, and Giannetti (2026), "Prompted to Start: How Generative AI is Transforming Entrepreneurship"](https://ssrn.com/abstract=5749564), the author built an LLM-based classification pipeline that identifies AI-native startups from Crunchbase data (descriptions, keywords, geography, founding year).
- This directness experiment is a separate strand of that work: having built the classifier, the natural next question is whether its verdicts are actually driven by the textual features it was shown, or by pretraining knowledge about the companies.
- Frame this as a general problem: as LLMs are increasingly used as measurement instruments in research, validating that their outputs are *direct* becomes essential.

### 3. Defining LLM Directness (~2 short paragraphs)

This is the conceptual heart of the README. Key beats:
- Introduce the concept from [Asirvatham, Mokski, and Shleifer (2026)](https://shleifer.scholars.harvard.edu/publication/gpt-measurement-tool) (NBER Working Paper 34834).
- Quote or closely paraphrase their definition: a measurement is *direct* when it is driven by the signal in the content itself, rather than by leakage, memorized facts, or correlated cues.
- Contrast with *indirect* measurement: an LLM that recognizes a company name and recalls facts about it from pretraining, then uses those facts rather than the input fields to produce a classification.
- Briefly note the two failure modes the paper identifies: (1) contamination/memorization, and (2) shortcut inference from correlated cues.

### 4. Experimental Design (~1-2 short paragraphs + table)

Describe the three-cell controlled experiment at a conceptual level:
- **Baseline**: all input features provided (name, descriptions, keywords, geography, founding year).
- **Arm A (feature ablation)**: real company name retained, but descriptions and keywords removed. Tests whether the classifier changes its verdict when substantive content is stripped.
- **Arm B (identity ablation)**: company name anonymized *and* descriptions/keywords removed. Tests whether the real name alone carries signal via pretraining memorization.

Include the existing table from the current README showing which fields are real vs. masked in each cell -- this is clean, timeless, and immediately communicable.

Briefly explain the interpretive logic:
- If Baseline and Arm A disagree, the classifier is direct (descriptions are doing the work).
- If they agree but Arm A and Arm B disagree for well-known companies, the classifier is leaking from the real name.

### 5. Statistical Methods (~1 short paragraph)

Brief, jargon-light summary that signals rigor. Mention:
- Chance-corrected agreement (Cohen's kappa)
- Paired hypothesis tests (McNemar's test for binary outcomes, Stuart-Maxwell for multi-class)
- Fame-stratified analysis (companies split into fame quartiles to test whether leakage concentrates among well-known firms)

No need to name sklearn or statsmodels -- those are implementation details.

### 6. Repository Overview (~5-8 lines, bullet list)

Very high-level, just enough to show this is real, structured code:
- `classify.py` -- pipeline CLI for running classification experiments
- `prompts/` -- controlled prompt files (byte-identical except for the experimentally varied input format)
- `scripts/` -- analysis scripts (agreement metrics, fame proxy, dashboard)
- `tests/` -- automated tests enforcing experimental controls (e.g., prompt consistency)

No file-by-file breakdown. No quickstart. No CLI flags.

### 7. References (~4-5 lines)

Cite the two papers:
- Bena, Bian, and Giannetti (2026) -- the companion classification paper
- Asirvatham, Mokski, and Shleifer (2026) -- the GABRIEL/directness paper (NBER WP 34834)

## What Gets Dropped from the Current README

The following sections from the [current README](README.md) will not carry over, as they are either operational detail or will go stale:
- Quickstart / CLI usage instructions
- Detailed repository layout (file-by-file)
- Prompt design details (the controlled-paragraph explanation)
- Threats to validity (too detailed for this audience)
- Reproducibility pinning
- "Why run the baseline ourselves" rationale

These remain available in the git history if needed, or could be moved to a separate `TECHNICAL.md` in the future if collaborators need them.
