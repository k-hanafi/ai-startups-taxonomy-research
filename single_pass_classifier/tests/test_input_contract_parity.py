"""Keep the standalone classifier and crawler CSV contracts aligned."""

from single_pass_classifier.input_contract import CLASSIFIER_INPUT_COLUMNS
from tavily_crawler.master_csv import CLASSIFIER_INPUT_COLUMNS as CRAWLER_INPUT_COLUMNS


def test_classifier_input_columns_match_crawler_output() -> None:
    assert CLASSIFIER_INPUT_COLUMNS == CRAWLER_INPUT_COLUMNS
