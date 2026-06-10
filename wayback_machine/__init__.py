"""Wayback Machine historical-evidence pipeline.

A self-contained sub-project that reconstructs the March-2023 (GPT-4 launch)
homepages of our classified startups from the Internet Archive and turns them
into a ``classifier_input_2023.csv`` that drops straight into the existing
classifier. It deliberately depends on nothing in ``src/`` (the one shared
cleaning function is vendored in :mod:`wayback_machine.evidence` and guarded by a
golden test) so the folder can be lifted into its own repository unchanged.

Run order: see README.md.
"""
