---
name: Taxonomy v2.1 Migration
overview: Migrate the taxonomy from 11 to 10 subclasses. Merge 0B and 0D into a single non-AI-native bucket. Promote 0C-THIN/THICK to AI-native and renumber the entire AI-native side so codes flow from foundation → tooling → wrappers → applied → autonomous → generative. RAD becomes a perfect function of ai_native.
todos:
  - id: update-prompt
    content: Rewrite prompts/Multiclassification_prompt.txt with new 1A–1G ordering, revised 0B (absorbs 0D), simplified RAD rules, OpenClaw added to 1F examples, all 10 few-shot examples renumbered
    status: completed
  - id: update-schema
    content: "Update src/schema.py subclass Literal to new 10-class set: 1A–1G, 0A, 0B, 0E"
    status: completed
  - id: update-merger
    content: Update src/merger.py subclass_order list to reflect new ordering
    status: completed
  - id: migration-script
    content: Write scripts/migrate_taxonomy_v21.py applying the full code remapping (1B→1E, 1C→1B, 1D→1F, 1E→1G, 0C-THIN→1C, 0C-THICK→1D, 0D→0B) and update ai_native for promoted wrappers
    status: completed
isProject: false
---

# Taxonomy v2.1 Migration Plan

## Summary of Design Changes

The new ordering groups codes by **how AI relates to the product mechanism**, flowing from infrastructure outward:

| New code | Class | New ai_native | Old code |
|---|---|---|---|
| 1A | Foundation Layer | 1 | 1A (unchanged) |
| **1B** | AI-Native Tooling & Ops | 1 | 1C |
| **1C** | Thin LLM Wrapper | 1 | 0C-THIN |
| **1D** | Thick LLM Integrator | 1 | 0C-THICK |
| **1E** | Applied Vertical AI | 1 | 1B |
| **1F** | Autonomous Agent Systems | 1 | 1D |
| **1G** | Generative Content Platforms | 1 | 1E |
| 0A | Traditional Tech / SaaS | 0 | 0A (unchanged) |
| **0B** | AI-Augmented / AI-Adjacent (merged) | 0 | 0B + 0D |
| 0E | Non-Tech | 0 | 0E (unchanged) |

**Total: 10 classes** (was 11). Removed: `0C-THIN`, `0C-THICK`, `0D`.

### Full code remapping for migration

| Old subclass | → | New subclass | ai_native change |
|---|---|---|---|
| `1A` | → | `1A` | none |
| `1B` (Vertical AI) | → | `1E` | none |
| `1C` (Tooling) | → | `1B` | none |
| `1D` (Agents) | → | `1F` | none |
| `1E` (Generative) | → | `1G` | none |
| `0A` | → | `0A` | none |
| `0B` | → | `0B` | none |
| `0C-THIN` | → | `1C` | **0 → 1** |
| `0C-THICK` | → | `1D` | **0 → 1** |
| `0D` | → | `0B` | none |
| `0E` | → | `0E` | none |

## RAD Rule — Now Fully Resolved

RAD is now a **perfect function of `ai_native`**. No exceptions, no sub-rules:

- **ai_native = 1** (all 7 classes) → always RAD-H, RAD-M, or RAD-L. Never RAD-NA.
- **ai_native = 0** (all 3 classes) → always RAD-NA. `conf_rad = null`.

This eliminates the former 0B/0D tension entirely. Because every ai_native=0 company — including old 0D ecosystem plays now living in merged 0B — gets RAD-NA, and because the only companies that ever received a real RAD score (old 0C-THIN/THICK) are now ai_native=1, the mapping is clean and lossless.

## RAD Rule — Now Fully Resolved

RAD is a **perfect function of `ai_native`**:

- `ai_native = 1` (1A–1G) → always RAD-H, RAD-M, or RAD-L
- `ai_native = 0` (0A, 0B, 0E) → always RAD-NA, `conf_rad = null`

No exceptions, no per-class sub-rules.

## Files to Change

### 1. [`prompts/Multiclassification_prompt.txt`](prompts/Multiclassification_prompt.txt)
The heaviest rewrite — the authoritative taxonomy for the LLM.

**AI-native subclasses, in new order:**
- `1A | Foundation Layer` — unchanged content
- `1B | AI-Native Infrastructure & Tooling` — unchanged content (was 1C)
- `1C | Thin LLM Wrapper` — content from old 0C-THIN, reframed as AI-native because GenAI IS the product mechanism
- `1D | Thick LLM Integrator` — content from old 0C-THICK, same reframing
- `1E | Applied Vertical AI` — unchanged content (was 1B)
- `1F | Autonomous Agent Systems` — unchanged content (was 1D), **add OpenClaw to the "Real examples" line** alongside Cognition (Devin), Adept, Dust, Browser Use, MultiOn
- `1G | Generative Content Platforms` — unchanged content (was 1E)

**Non-AI-native subclasses:**
- `0A | Traditional Tech / SaaS` — unchanged
- `0B | AI-Augmented / AI-Adjacent` — rewritten to encompass both old 0B (AI as a non-core feature) and old 0D (ecosystem/adjacent benefit without AI in the product). Key shared trait: GenAI is not the product mechanism.
- `0E | Non-Tech` — unchanged

**RAD assignment rules:** Replace all per-class rules with the simple bidirectional rule above.

**RAD-NA definition:** Update to apply to all `ai_native = 0` classes (0A, 0B, 0E).

**Edge cases:** Remove all `0C-*` and `0D` references; update the LLM-wrapper edge case to point to 1C/1D; update the human-in-the-loop labeling case from 0D to 0B.

**Few-shot examples (all 10 need renumbering):**

| # | Company | Old subclass | New subclass | ai_native change |
|---|---|---|---|---|
| 1 | Pinecone | 1A | 1A | — |
| 2 | DeepScribe | 1B | **1E** | — |
| 3 | Braintrust | 1C | **1B** | — |
| 4 | Cognition Labs | 1D | **1F** | — |
| 5 | ElevenLabs | 1E | **1G** | — |
| 6 | Perplexity AI | 0C-THICK | **1D** | **0 → 1** |
| 7 | SummarizeIt | 0C-THIN | **1C** | **0 → 1** |
| 8 | Shipsi | 0B | 0B | — |
| 9 | CoreWeave | 0D | **0B** | — |
| 10 | Harvey AI | 1B | **1E** | — |

For Example 6 (Perplexity) and Example 7 (SummarizeIt): change `ai_native: 0` to `ai_native: 1`. Their existing RAD scores remain valid. Section header text and reasoning prose should reference the new codes (e.g., "1D not 1E" instead of "0C-THICK not 1B").

### 2. [`src/schema.py`](src/schema.py)
Single source of truth for the JSON schema injected into every batch request.

```python
subclass: Literal[
    "1A", "1B", "1C", "1D", "1E", "1F", "1G",
    "0A", "0B", "0E",
]
```

### 3. [`src/merger.py`](src/merger.py)
Update `subclass_order` (line 84):

```python
subclass_order = [
    "1A", "1B", "1C", "1D", "1E", "1F", "1G",
    "0A", "0B", "0E",
]
```

### 4. CSV migration script (`scripts/migrate_taxonomy_v21.py`)
The existing `outputs/classified_startups_v2.csv` (265k rows) needs the **full** code remapping, not just the merged/promoted rows, because every AI-native code shifts position.

```python
SUBCLASS_REMAP = {
    "1A": "1A",
    "1B": "1E",   # Vertical AI moved
    "1C": "1B",   # Tooling moved
    "1D": "1F",   # Agents moved
    "1E": "1G",   # Generative moved
    "0A": "0A",
    "0B": "0B",
    "0C-THIN":  "1C",   # promoted to AI-native
    "0C-THICK": "1D",   # promoted to AI-native
    "0D": "0B",   # merged
    "0E": "0E",
}
PROMOTED_TO_NATIVE = {"0C-THIN", "0C-THICK"}  # need ai_native flipped 0→1
```

The script:
- Applies `SUBCLASS_REMAP` to every row
- Sets `ai_native = 1` for rows whose **old** subclass was in `PROMOTED_TO_NATIVE`
- Leaves `rad_score`, `conf_rad`, and all reasoning fields untouched (they remain valid: old 0C-THIN/THICK already had real RAD-H/M scores; old 0D already had RAD-NA; both still correct under the new rule)
- Writes to `outputs/classified_startups_v21_migrated.csv` (original preserved)

This migration is lossless for RAD and gives you a usable v2.1 dataset for analysis without re-running the LLM pipeline.

## Execution Order

1. Update prompt → schema → merger (in that order; prompt first establishes the truth)
2. Optionally run the CSV migration script on old data for immediate exploratory analysis
3. Re-run the full pipeline (`classify.py run`) to produce fresh v2.1 LLM-classified results
