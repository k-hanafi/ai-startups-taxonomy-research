"""Tests for the tiktoken-based cost estimator.

Verifies that token counts are accurate and cost projections are internally
consistent. All tests run offline with no API calls.
"""

from src.tokens import (
    BATCH_DISCOUNT,
    CACHE_DISCOUNT,
    CostEstimate,
    count_tokens,
    estimate_cost,
    get_encoding,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_string(self):
        tokens = count_tokens("Hello, world!")
        assert 1 <= tokens <= 10

    def test_longer_string_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("This is a significantly longer sentence with many words.")
        assert long > short


class TestGetEncoding:
    def test_fallback_on_unknown_model(self):
        enc = get_encoding("nonexistent-model-xyz")
        assert enc is not None

    def test_returns_encoding_object(self):
        enc = get_encoding("gpt-5.4-nano")
        assert hasattr(enc, "encode")


class TestEstimateCost:
    def _make_estimate(self, n: int = 100) -> CostEstimate:
        prompt = "You are a classifier. Classify the following startup."
        messages = [f"CompanyID: test-{i}\nCompanyName: Test {i}" for i in range(n)]
        return estimate_cost(prompt, messages, model="gpt-5.4-nano", batch_size=50)

    def test_company_count(self):
        est = self._make_estimate(100)
        assert est.total_companies == 100

    def test_batch_count(self):
        est = self._make_estimate(100)
        assert est.batches_needed == 2  # 100 / 50

    def test_batch_count_partial(self):
        est = self._make_estimate(75)
        assert est.batches_needed == 2  # ceil(75 / 50)

    def test_prefix_tokens_positive(self):
        est = self._make_estimate()
        assert est.system_prompt_tokens > 0
        assert est.schema_tokens > 0
        assert est.prefix_tokens == est.system_prompt_tokens + est.schema_tokens

    def test_total_input_greater_than_prefix(self):
        est = self._make_estimate()
        assert est.total_input_tokens > est.prefix_tokens

    def test_batch_discount_applied(self):
        est = self._make_estimate()
        assert est.cost_total_batch < est.cost_total_sync
        ratio = est.cost_total_batch / est.cost_total_sync
        assert abs(ratio - BATCH_DISCOUNT) < 0.01

    def test_caching_cheaper_than_batch(self):
        est = self._make_estimate()
        assert est.cost_with_caching <= est.cost_total_batch

    def test_format_report_runs(self):
        est = self._make_estimate()
        report = est.format_report()
        assert "PRE-FLIGHT COST ESTIMATE" in report
        assert "gpt-5.4-nano" in report
        assert "$" in report
