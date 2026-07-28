from __future__ import annotations

from two_pass_classifier.prompts import (
    BINARY_GATE_PROMPT,
    FAMILY_BLOCK_AI,
    FAMILY_BLOCK_NOT_AI,
    SUBCLASS_RAD_PROMPT,
    load_pass_a_prompt,
    load_pass_b_prompt,
)


def test_pass_a_owns_ai_reasoning_only():
    prompt = load_pass_a_prompt()
    assert "ai_native_reasoning" in prompt
    assert "ai_native_critique" in prompt
    assert "subclass_reasoning" not in prompt
    assert "rad_reasoning" not in prompt


def test_pass_b_owns_subclass_and_rad_reasoning_only():
    for family in (0, 1):
        prompt = load_pass_b_prompt(family)
        assert "subclass_reasoning" in prompt
        assert "rad_reasoning" in prompt
        assert "ai_native_reasoning" not in prompt
        assert "{family_block}" not in prompt


def test_family_prompts_have_only_their_strict_output_fields():
    ai = load_pass_b_prompt(1)
    not_ai = load_pass_b_prompt(0)

    assert "1A | Foundation Layer" in ai
    assert "0A | Traditional Tech or SaaS" not in ai
    assert "rad_confidence" in ai
    assert "rad_critique" in ai

    assert "0A | Traditional Tech or SaaS" in not_ai
    assert "1A | Foundation Layer" not in not_ai
    assert "Do not output rad_score, rad_confidence, or rad_critique" in not_ai


def test_prompt_files_have_one_package_source():
    for path in (
        BINARY_GATE_PROMPT,
        SUBCLASS_RAD_PROMPT,
        FAMILY_BLOCK_AI,
        FAMILY_BLOCK_NOT_AI,
    ):
        assert path.parent.name == "prompts"
        assert path.parent.parent.name == "two_pass_classifier"
        assert path.is_file()
