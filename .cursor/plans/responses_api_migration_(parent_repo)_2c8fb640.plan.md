---
name: Responses API migration (parent repo)
overview: "Apply the same Chat Completions → Responses API migration already done in `llm_directness_experiment` to `ai-native-startup-classification`: batch JSONL shape, batch endpoint, downloader parsing, and synchronous `classify.py test`. No URL scraping or formatter expansion—those remain future optional work on the parent pipeline only."
todos:
  - id: builder-responses
    content: "Update ai-native src/builder.py: responses_text_format_json_schema + build_request_body for POST /v1/responses"
    status: completed
  - id: submitter-endpoint
    content: Set batches endpoint to /v1/responses in src/submitter.py
    status: completed
  - id: downloader-dual-parse
    content: Port dual batch body + usage parsing from directness src/downloader.py
    status: completed
  - id: classify-test
    content: Switch classify.py _cmd_test to client.responses.create + output_text
    status: completed
  - id: pyproject-openai
    content: Bump pyproject.toml openai to >=2.0.0; run pytest
    status: completed
  - id: git-commit
    content: Single git commit in taxonomy repo per coding_instructions.md Git Conventions (after pytest green)
    status: completed
isProject: false
---

# Migrate `ai-native-startup-classification` to Responses API

## Context

[`ai-native-startup-classification`](/Users/k/Desktop/ai-native-startup-classification/) is the **main v2 taxonomy pipeline** (single prompt, no arms). It currently matches what `llm_directness_experiment` looked like **before** the Responses migration:

| Location | Current behavior |
|----------|------------------|
| [`src/builder.py`](/Users/k/Desktop/ai-native-startup-classification/src/builder.py) | `POST /v1/chat/completions`, `messages`, `response_format`, `max_completion_tokens` |
| [`src/submitter.py`](/Users/k/Desktop/ai-native-startup-classification/src/submitter.py) | `endpoint="/v1/chat/completions"` |
| [`src/downloader.py`](/Users/k/Desktop/ai-native-startup-classification/src/downloader.py) | Parses only `choices[0].message.content` and Chat Completions `usage` |
| [`classify.py`](/Users/k/Desktop/ai-native-startup-classification/classify.py) `_cmd_test` | `client.chat.completions.create` + `choices[0].message.content` |

This plan **ports the same edits** already applied in [`llm_directness_experiment/src/builder.py`](/Users/k/Desktop/llm_directness_experiment/src/builder.py), [`submitter.py`](/Users/k/Desktop/llm_directness_experiment/src/submitter.py), [`downloader.py`](/Users/k/Desktop/llm_directness_experiment/src/downloader.py), and [`classify.py`](/Users/k/Desktop/llm_directness_experiment/classify.py).

## Out of scope (explicit)

- **No website URL scraping or formatter upgrades** — your note that the parent repo may later scrape company sites is acknowledged; this change does not add URLs to the user message or CSV. The directness experiment repo intentionally stays without that baseline upgrade.
- **No README / `coding_instructions.md` edits** unless you ask — optional follow-up is a one-line note that batch jobs use `POST /v1/responses`.

## Implementation steps

### 1. [`src/builder.py`](/Users/k/Desktop/ai-native-startup-classification/src/builder.py)

- Update the module docstring: batch lines target **Responses API**, not “chat completion”.
- Add `responses_text_format_json_schema(schema: dict) -> dict` returning the `text` fragment with `format.type == "json_schema"`, `name: "ClassificationResult"`, `strict: True`, and the Pydantic schema (same pattern as directness repo).
- Replace `build_request_body` body with:
  - `"url": "/v1/responses"`
  - `"body"`: `model`, `instructions` (= system prompt), `input` (= user message string), `prompt_cache_key` (keep existing [`PROMPT_CACHE_KEY`](/Users/k/Desktop/ai-native-startup-classification/src/config.py) value `"v2-classifier-system-prompt"`), `max_output_tokens` ([`MAX_OUTPUT_TOKENS`](/Users/k/Desktop/ai-native-startup-classification/src/config.py)), `store: False`, `text: responses_text_format_json_schema(schema)`.

### 2. [`src/submitter.py`](/Users/k/Desktop/ai-native-startup-classification/src/submitter.py)

- Change `client.batches.create(..., endpoint="/v1/responses", ...)`.

### 3. [`src/downloader.py`](/Users/k/Desktop/ai-native-startup-classification/src/downloader.py)

Mirror the directness repo helpers:

- `_assistant_json_from_batch_body(body)`: prefer Responses `output` → `output_text` blocks; fallback to legacy `choices[0].message.content` so **already-downloaded** old batch JSONL remains parseable.
- `_usage_from_batch_body(body)`: map Responses `input_tokens` / `output_tokens` / `input_tokens_details.cached_tokens` to the existing internal keys (`prompt_tokens`, `completion_tokens`, `cached_tokens`) used downstream for [`merger.py`](/Users/k/Desktop/ai-native-startup-classification/src/merger.py) cost reporting.
- Refactor `_parse_result_line` to use these helpers.

### 4. [`classify.py`](/Users/k/Desktop/ai-native-startup-classification/classify.py) — `_cmd_test`

- Import `responses_text_format_json_schema` from `src.builder`, and `MAX_OUTPUT_TOKENS`, `PROMPT_CACHE_KEY` from `src.config`.
- Replace `client.chat.completions.create` with `client.responses.create(...)` using `instructions`, `input`, `prompt_cache_key`, `max_output_tokens`, `store=False`, `text=responses_text_format_json_schema(schema)`, `service_tier` unchanged.
- Parse JSON with `json.loads(response.output_text)` (SDK helper on [`Response`](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py)).
- Optionally add one sentence to the top-of-file docstring that `test` uses the Responses API.

### 5. Dependency pin — [`pyproject.toml`](/Users/k/Desktop/ai-native-startup-classification/pyproject.toml)

- Bump `openai>=1.75.0` to **`openai>=2.0.0`** so `client.responses` and `Response.output_text` match the surface used in practice (aligned with a typical current install).

### 6. Verification

- Run `pytest` in `ai-native-startup-classification` after edits.
- **Operational note** (for you, not code): any **new** batch run must use freshly generated JSONL from `prepare` after this merge; in-flight Chat Completions batches are unaffected. Re-run `prepare` before full production batch if old JSONL still points at `/v1/chat/completions`.

### 7. Git commit ([`coding_instructions.md`](/Users/k/Desktop/ai-native-startup-classification/coding_instructions.md) § Git Conventions)

Make **one milestone commit** after implementation is complete and tests pass. Do not split into per-file commits. Optional branch: `feat/responses-api-batch-pipeline` (or `refactor/…` if you treat this as API-shape alignment).

**Title** (imperative mood, value not mechanics, under 72 characters). Example:

`Migrate batch pipeline to OpenAI Responses API`

**Body** (1–3 sentences, explain why and constraints; follow project prose rules: no em dashes, no semicolons to chain clauses, short sentences):

```
Migrate batch pipeline to OpenAI Responses API

OpenAI recommends Responses for new work. Structured outputs use text.format
instead of response_format. Batch jobs and classify.py test share the same
envelope. Downloader still parses legacy Chat Completions batch lines so old
downloads remain usable. openai bumped to 2.x for client.responses.
```

Adjust the body if the final diff differs (for example mention dual-parse only if retained). The title should answer what the project gains: aligned with platform direction, consistent with the directness experiment repo, preserved backward compatibility for downloaded artifacts.

## Architecture note (future URL work)

If you later add scraped web text to the parent pipeline, changes will land in [`src/formatter.py`](/Users/k/Desktop/ai-native-startup-classification/src/formatter.py) (and possibly CSV columns)—the Responses request envelope (`instructions` + `input` + `text.format`) stays the same unless OpenAI changes the API again.
