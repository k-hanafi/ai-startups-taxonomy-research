"""Golden-set evaluation harness for the startup classifier.

The harness owns golden data, matrix orchestration, scoring, calibration,
dashboards, and archives. ``two_pass_classifier`` owns every classifier
contract used by the harness: prompts, schemas, formatting, cohort assignment,
request bodies, confidence extraction, supported models, output caps, and
pricing.
"""
