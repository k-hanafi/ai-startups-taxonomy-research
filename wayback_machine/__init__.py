"""Wayback Machine historical-evidence pipeline.

A self-contained sub-project that reconstructs the March-2023 (GPT-4 launch)
homepages of our classified startups from the Internet Archive and turns them
into a ``classifier_input_2023.csv`` that drops straight into the existing
classifier. Its evidence-recovery pipeline stays independent of root application
packages. The shared cleaner is vendored in :mod:`wayback_machine.evidence` and
guarded by a golden test. :mod:`wayback_machine.classify_2023` is the intentional
namespaced bridge to the unchanged V1 classifier.

Run order: see README.md.
"""
