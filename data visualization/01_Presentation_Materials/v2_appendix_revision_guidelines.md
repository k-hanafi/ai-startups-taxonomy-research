# V2 Appendix Revision Guidelines

Professor-facing notes for revising Appendix B.2 so it matches the current V2 classification method. This is a revision checklist, not finished appendix prose. Prefer short bullets when rewriting.

**Audience note:** Keep the appendix about the research method, sample construction, taxonomy, confidence, results, and limitations. Do not describe software tooling, code, or operational infrastructure.

---

## 1. Quick revision map

### Keep
- Core AI-native definition: AI must be central to the product, service, production technology, or business model, not just an internal tool or marketing claim.
- The same 10 mutually exclusive taxonomy categories (7 AI-native, 3 not AI-native).
- RAD as a structural reliance score for AI-native firms only: RAD-H / RAD-M / RAD-L, and RAD-NA otherwise.
- Use of Crunchbase fields plus website evidence as the classifier input.
- Binary AI-native indicator plus finer subclass and RAD heterogeneity.

### Replace
- “Two-stage Tavily-enriched LLM pipeline” as if Tavily and the LLM were one joint classifier.
  - Prefer: **evidence construction**, then a **two-pass LLM classifier**.
- Model claim: V1 used `gpt-5.4-nano` (May 5, 2026).
  - Prefer: production V2 uses **`gpt-5.6-luna`**. Use the completed production classification date once the full run is finalized.
- Evidence claim: “web-crawl text retrieved through the Tavily Search API.”
  - Prefer: Tavily **website crawl** of up to five selected pages per company, cleaned of navigation and other low-information text.
- Confidence claim: one overall confidence score.
  - Prefer: Pass A probability-based AI-native confidence, plus separate 1–5 subclass and RAD confidence ratings.
- Sample claim: all 44,387 Crunchbase firms are the classified V2 universe.
  - Prefer: V2 classifies the **evidence-based sample of 37,746** firms with usable website text.

### Add
- Wayback Machine recovery for firms whose current websites could not be retrieved.
- Explicit survivorship-bias motivation and recovery funnel counts.
- Two-pass design rationale: binary gate first, then subclass and RAD within that fixed family.
- Log-probability confidence method and how it should be interpreted.
- Golden-set validation metrics (accuracy, calibration, selective accuracy).
- Snapshot-date and live-versus-archived evidence provenance.

### Remove or avoid
- Claims that high average confidence proves low hallucination risk.
- Claims that every firm has five pages of website evidence.
- Claims that “live” means verified legal operating status.
- Conflating the production Wayback recovery with the separate paused March-2023 historical study.
- Software/system details (commands, files, retries, caching, rate limits, run IDs).

---

## 2. Research sample and evidence construction

### What the revised appendix should say
- Starting sample: **44,387** startups from Crunchbase.
- Structured fields used with website evidence: short and long company descriptions, category keywords, founding date, employee count, and funding.
- Website evidence comes from a Tavily crawl that selects **up to five** pages most informative about the product or business model.
- Extracted page text is cleaned to remove navigation, legal boilerplate, and other low-information content. Very short residual text is treated as missing evidence.
- Final V2 analysis sample: firms with usable website evidence only.

### Key statistics
| Item | Count |
|------|------:|
| Crunchbase starting sample | 44,387 |
| Firms with usable live website evidence | 22,032 |
| Firms with usable recovered archived evidence | 15,714 |
| Final evidence-based V2 sample | 37,746 |
| Starting sample without usable website evidence after live crawl | 22,355 |
| Of those, firms with a usable website host for archive recovery | 22,002 |

### Qualifications
- “Up to five pages” is a crawl budget, not a guarantee. Some firms contribute fewer usable pages after cleaning.
- Live versus archived labels the **evidence source**, not a verified legal alive/dead status.
- Website evidence is the primary signal of what the company builds and sells. Crunchbase fields are supporting context and can be stale or generic.

---

## 3. Wayback Machine recovery and survivorship bias

### Why this matters
- Restricting the sample to currently crawlable websites would systematically under-represent firms that disappeared, rebranded, parked domains, or otherwise lost retrievable live sites.
- The V2 sample therefore recovers pre-disappearance website evidence from the Internet Archive whenever possible.

### Method in plain language
1. Identify firms with no usable live website evidence after the Tavily crawl.
2. Use the firm’s last archived website capture as a proxy for disappearance timing.
3. Seek an earlier archived page from roughly **six months before** that last capture.
4. Extract and clean the archived page text with the same usefulness rules used for live evidence.
5. Include recovered firms in the evidence-based sample alongside live-evidence firms.

### Recovery funnel
| Step | Count |
|------|------:|
| Missing-evidence firms with a usable host | 22,002 |
| Firms with a usable archive target | 19,044 |
| Firms with usable recovered website evidence | 15,714 |

### Qualifications
- Archive coverage is incomplete. Not every missing-evidence firm can be recovered.
- The “disappearance date” is an archive proxy, not a formal business-closure date.
- Archived evidence is usually shallower than the live five-page crawl. Current recovery is primarily homepage-based, while live evidence can include up to five selected pages.
- This production recovery is **not** the separate March-2023 historical study. That historical strand uses a common GPT-4-launch archive window to study messaging change and remains paused. Do not describe V2 as classifying every firm on a March-2023 homepage.

---

## 4. Two-pass AI-native classification

### Preferred framing
Rewrite Stage 1 / Stage 2 as:

1. **Evidence construction:** Crunchbase fields + live or recovered website text.
2. **Two-pass LLM classification:**
   - **Pass A:** decide only whether the firm is AI-native.
   - **Pass B:** holding that family fixed, assign the mutually exclusive subclass and RAD category.

### Research rationale
- The binary AI-native decision is the primary research construct.
- Separating it from subclass and RAD reduces task complexity.
- Pass B cannot reverse the Pass A family decision, so finer categories cannot quietly change the main indicator.

### Model and assignment rules
- Production model: **`gpt-5.6-luna`**.
- Classification date: insert the finalized production-run date once complete.
- Cohort (pre-GenAI versus GenAI-era) is assigned from founding date relative to the GPT-4 launch boundary (**2023-03-14**), not inferred by the model.
- The classifier still returns:
  - binary AI-native indicator
  - mutually exclusive taxonomy category
  - RAD score for AI-native firms
  - confidence measures
  - short reasoning and sources used

### Taxonomy to retain
Keep the V1 category meanings. Suggested plain labels:

**AI-native**
- Foundation layer
- AI-native infrastructure and tooling
- Thin LLM wrapper
- Thick LLM integrator
- Applied vertical AI
- Autonomous agent systems
- Generative content platforms

**Not AI-native**
- Traditional tech / SaaS
- AI-augmented
- Non-tech

### RAD meanings to retain
- **RAD-H:** core product depends structurally on third-party models or APIs.
- **RAD-M:** uses external AI services, but also has meaningful proprietary workflow, data, or infrastructure.
- **RAD-L:** relies mainly on in-house models, proprietary infrastructure, or AI capabilities less directly tied to third-party APIs.
- **RAD-NA:** not AI-native, so RAD does not apply.

---

## 5. Confidence and validation

### What Pass A confidence is
- Pass A confidence is the probability the model assigned to its chosen AI-native label (`0` or `1`).
- It is estimated from the model’s token probabilities over the most likely alternatives for that decision.
- If the opposing label is missing from those alternatives and the resulting uncertainty band is too wide, confidence is left blank rather than forced.

### What Pass B confidence is
- Subclass confidence and RAD confidence are separate **1–5** ratings produced in Pass B.
- They are not the same as Pass A’s probability-based score.

### How to interpret confidence correctly
- Low Pass A confidence can flag an uncertain binary classification.
- High average confidence does **not** by itself prove low hallucination risk.
- Why: the score measures certainty about the binary label, not whether every factual claim in the explanation is true.
- Analogy for the appendix or a footnote: high confidence is closer to “the model strongly preferred this answer” than to “the model cannot be wrong.”

### Recommended validation language
On a labeled golden set of **100** companies, report whether confidence tracks correctness using:
- AI-native accuracy with a confidence interval
- confidence coverage (share of firms with a usable Pass A confidence score)
- expected calibration error (ECE)
- the gap between mean confidence and accuracy
- selective accuracy on the most confident half of predictions

### Current golden-set evidence for the production model
Latest archived real golden-set matrix in the project materials (100 labeled companies, provisional gold labels):

| Metric | `gpt-5.6-luna` |
|--------|---------------:|
| AI-native accuracy | 0.91 |
| Mean Pass A confidence | 0.989 |
| ECE | 0.079 |
| Selective accuracy on most confident 50% | 0.96 |
| Confidence coverage | 100 / 100 |

Interpretation for the appendix:
- Confidence is available for the full golden set.
- Accuracy is high, and the most confident half is more accurate than the full set.
- Mean confidence still exceeds accuracy, so the model is somewhat overconfident.
- Therefore, confidence is useful as an uncertainty and calibration signal, not as a direct hallucination rate.

### If the paper wants a hallucination claim
Do **not** infer it from average log-probability confidence alone. Add a separate manual evidence-grounding audit, for example checking whether key product claims in the model reasoning appear in the website or Crunchbase evidence.

---

## 6. Research outputs and summary statistics

### Outputs the appendix should mention
- AI-native indicator
- Taxonomy subclass
- RAD score
- Pass A AI-native confidence
- Pass B subclass and RAD confidence
- Short reasoning / critique
- Sources used
- Evidence source: live website versus recovered archive
- Website snapshot date

### Stable sample-construction statistics
Use the counts in Sections 2 and 3 now.

### Pending final production classification statistics
Fill only after the complete V2 production output exists. Do not use partial-run distributions.

Placeholder table:

| Statistic | Value |
|-----------|------:|
| Production model | `gpt-5.6-luna` |
| Classification completion date | _pending_ |
| Final classified N | _pending; target 37,746_ |
| Share AI-native | _pending_ |
| Subclass distribution | _pending_ |
| RAD distribution among AI-native firms | _pending_ |
| Results by cohort | _pending_ |
| Results by live versus archived evidence | _pending_ |
| Pass A confidence coverage in production | _pending_ |
| Pass A confidence distribution | _pending_ |
| Pass B confidence distributions | _pending_ |

---

## 7. Limitations to surface

- The V2 analysis sample is evidence-based, not the full Crunchbase extract.
- Some firms remain missing because neither live crawl nor archive recovery produced usable website text.
- Live evidence can be deeper than archived evidence.
- Archive snapshot dates vary across firms and years.
- Disappearance timing is inferred from archive history.
- The classifier can still err, including when confidence is high.
- Golden-set labels used for validation are provisional and should be described as such until finalized.
- Confidence calibration is about label correctness, not automatic proof against hallucinated explanations.

---

## 8. Suggested appendix section order

1. Research objective and AI-native definition
2. Sample and evidence construction
3. Wayback recovery and survivorship bias
4. Two-pass classification procedure
5. Taxonomy and RAD definitions
6. Confidence measures and golden-set validation
7. Output fields and summary statistics
8. Limitations

---

## 9. Phrases to prefer / avoid

| Prefer | Avoid |
|--------|-------|
| Evidence construction, then a two-pass LLM classifier | Two-stage Tavily-enriched LLM pipeline |
| Up to five selected website pages | Five pages of website evidence for every firm |
| Live versus archived evidence source | Live means the company is still operating |
| Probability assigned to the chosen AI-native label | High confidence proves low hallucination |
| Recovered pre-disappearance archive evidence | All firms classified on March 2023 homepages |
| Evidence-based sample of 37,746 | Full 44,387-firm classification universe for V2 |
| `gpt-5.6-luna` | `gpt-5.4-nano` as the current production model |
