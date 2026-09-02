"""Preloaded demo corpus.

Six projects from the evaluation benchmark, chosen because the two systems
scored them differently. Between them they cover a clean unanimous round, a
round the frontier model rejects, and a document the pipeline abstains on.
Provenance: paper/data/misclassifications.csv
"""

PROJECTS = [
    {
        "rera_id": "WBRERA/P/SOU/2023/000294",
        "label": "West Bengal / South 24 Parganas",
        "note": "Clean case: votes agree on the first round.",
    },
    {
        "rera_id": "WBRERA/P/SOU/2023/000155",
        "label": "West Bengal / Prestige",
        "note": "Has no OC or CC. The predecessor wrote a GST certificate in as an Occupancy Certificate dated 2022-10-19.",
    },
    {
        "rera_id": "MAA07612/A2M/EX1/020725/310826",
        "label": "Gujarat / Ahmedabad",
        "note": "Misrouted by both systems, in opposite directions.",
    },
    {
        "rera_id": "UPRERAPRJ757529",
        "label": "Uttar Pradesh",
        "note": "Still false-accepted. Shown deliberately.",
    },
    {
        "rera_id": "HIRA/P/SOU/2018/000168",
        "label": "West Bengal / HIRA",
        "note": "Multi-document project with an extension certificate.",
    },
    {
        "rera_id": "RERA-GRG-100-2018",
        "label": "Haryana / Gurugram",
        "note": "Predecessor returned 28/02/19 under an inferred date format.",
    },
]
