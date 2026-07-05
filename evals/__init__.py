"""Golden-set evaluation harness for the startup classifier.

Standalone package: benchmarks OpenAI models against a 100-company
human-verified golden dataset and validates the logprob-confidence
methodology before any production pipeline change.

Only three read-only production-identity artifacts may be imported from
src/ (schema, formatter, system prompt loader). Nothing here modifies src/.
"""
