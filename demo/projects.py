"""Preloaded demo corpus.

Five projects from the evaluation benchmark. Between them they cover a full
certificate lifecycle, two straightforward Occupancy Certificate cases, and two
projects the predecessor pipeline got wrong in different ways.

Ground truth and baseline outcomes: paper/data/generator_verifier_baseline.csv
The UI renders `rera_id` only; `label` is documentation.
"""

PROJECTS = [
    {
        "rera_id": "P01100001916",
        "label": "Telangana",
    },
    {
        "rera_id": "HIRA/P/KOL/2019/000445",
        "label": "West Bengal / Kolkata (HIRA)",
    },
    {
        "rera_id": "WBRERA/P/SOU/2023/000024",
        "label": "West Bengal / South 24 Parganas",
    },
    {
        "rera_id": "WBRERA/P/SOU/2023/000215",
        "label": "West Bengal / South 24 Parganas",
    },
    {
        "rera_id": "RERA-GRG-100-2018",
        "label": "Haryana / Gurugram",
    },
]
