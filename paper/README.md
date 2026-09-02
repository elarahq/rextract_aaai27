# AAAI-27 Demonstration Track submission

**Deadline: 18 September 2026, 11:59pm AOE.** Notification 6 Nov 2026,
camera-ready 20 Nov 2026, demos presented 18–21 Feb 2027, Montréal.

Submit at: <https://openreview.net/group?id=AAAI.org/2027/Demonstration_Program>

## Build

```bash
latexmk -pdf main.tex
```

The call requires **2 pages of content + 1 page of references only**.
After the data update the content runs ~110 words long, spilling a short
block onto page 3 above the references. **Trimming is outstanding and
left to the author.** Verify with:

```bash
latexmk -C && latexmk -pdf main.tex
```

then confirm page 3 holds nothing but references. If a build ever looks
doubled, that's stale artifacts — `latexmk -C` clears it. `xelatex` and
`lualatex` both fail here; `aaai2027.sty` is pdfLaTeX only.

## Files

| File | Notes |
|---|---|
| `main.tex` | The paper. System name is one macro: `\newcommand{\sys}{RE-XTRACT}` |
| `refs.bib` | 8 references, all real and verifiable |
| `aaai2027.sty`, `aaai2027.bst` | Official AuthorKit27 files, unmodified |
| `paper/data/classification_eval.csv` | Per-class P/R/F1, both systems — source for Table 1 |
| `paper/data/confusion_matrices.csv` | 4×4 confusion matrix per system — source for Table 2 |
| `paper/data/misclassifications.csv` | Every misclassified document, both systems (21 + 7, reconciles with the matrices) — source for the demo scenarios |
| `paper/data/generator_verifier_baseline.csv` | Baseline per-project extraction outcomes, 25 projects (not yet in the paper) |

Do **not** add `\bibliographystyle` — `aaai2027.sty` sets it, and adding
it again is a BibTeX error. `hyperref` is forbidden by the style.

Currently set to **single-blind** (author names visible), which the call
permits and which lets the production-deployment context work for you.
For double-blind, change `\usepackage{aaai2027}` to
`\usepackage[submission]{aaai2027}`.

## Before you submit

**1. Add co-authors.** Only Harshul Kuhar is listed; there's a commented
placeholder in `main.tex`.

**2. Numbers are traceable now — keep them that way.** Everything in
Tables 1 and 2 comes from the two evaluation reports recorded in `paper/data/`,
both scored against the same ground truth
(`classifier_testing/current/inputlinka.csv`) over the same 143 documents:

| | Gen.–verifier | RE-XTRACT |
|---|---|---|
| report date | 2026-04-17 | 2026-04-20 |
| predictions | `classification_resultspure10.csv` | `classification_results.csv` |
| accuracy | 85.3% (122/143) | 95.1% (136/143) |
| macro-F1 | .804 | .955 |
| silent errors | 21 | 2 |

Both confusion matrices reconcile exactly — row sums 45/8/62/28,
diagonals 122 and 136. Three claims from the earlier draft were
**falsified** by this data and are gone: 95.7% accuracy, an 88.7%
baseline, and "100% recall on all four in-schema types" (actual: RC .98,
EC 1.00, OC/CC .94).

**2a. Date-extraction data still pending** — 71 documents, per the author.
The paper currently claims nothing about extraction accuracy, only
routing, so no placeholder needs removing. When it lands it wants a third
table. `paper/data/generator_verifier_baseline.csv` already holds the
baseline's per-project extraction outcomes (25 projects, 3 states) and
its error taxonomy — wrong date format (MM/DD/YY), wrong value, duplicate
document with conflicting dates, missed CC, GST certificate written as
Occupancy. Each maps onto a specific gate, which argues the design better
than an accuracy delta does. Caveat: that sheet's own summary doesn't
close (68 + 3 = 71 against a stated 73 relevant) — resolve before use.

**3. The demo described here does not exist yet.** This is the real work
item. The paper describes a web interface with a live vote panel, an
ablation console, and a provenance viewer. Today the system is a CLI
(`main.py`). Everything the interface renders is already in the audit log
— it's a rendering job, not new logic — but it has to be built, because:

- The 5-minute video has to show it, and the video is submitted *with*
  the paper on 18 Sept.
- If accepted, you stand at a table with it running on 18–21 Feb 2027.

**4. Demo scenarios are now grounded in real runs.** This was previously
flagged as a reconstruction; it no longer is. All four scenarios in the
paper come from `paper/data/misclassifications.csv`, cross-referenced between
the two evaluation reports:

| Scenario | Document | Verified fact |
|---|---|---|
| The common case | a WB Registration Certificate | unanimous round 1 + validator confirm, as logged by the pipeline at run time |
| The fix | `WBRERA/P/SOU/2023/000155` | baseline false-accepted **5** documents here, incl. a GST cert written as Occupancy dated 2022-10-19; RE-XTRACT gets all 5 right |
| The flip | `MAA07612/A2M/EX1/020725/310826` | wrong in **both** systems, opposite directions: baseline false-accept, RE-XTRACT conservative reject |
| The honest failure | `UPRERAPRJ757529` | false-accepted by both; shown deliberately |

The invented "Gujarati Building Use Permission" scenario has been removed.
Two things still need doing: pull the actual PDFs for these four projects
into the demo corpus, and confirm the round-by-round behaviour matches
(the routing labels are verified, the *per-round* vote sequence for the
first scenario is verified only for a WB Registration Certificate
generally, not that specific document).

Also worth knowing for the video: 8 projects the baseline got wrong are
fully fixed, 3 are still wrong, and 1 (`UPRERAPRJ918`) is newly wrong in
RE-XTRACT only — a conservative reject. That 8/3/1 split is honest
framing if a reviewer asks what regressed.

**5. Record the video.** Up to 5 minutes, submitted as supplementary
material in OpenReview. The call says videos are weighted more heavily
than slides. Recommended: 30–60s overview up front, then drive the three
scenarios in order. End on the abstention — a system that visibly refuses
to answer is the memorable moment and is the actual contribution.

**6. Internal host is no longer hardcoded.** `OLLAMA_API_URL` now reads from
the environment and defaults to localhost, so the source carries no internal
network address. The real value lives in `.env`, which `.gitignore` excludes.

## Relationship to `paper/`

`paper/` holds the longer IEEE-format manuscript (8 pages, full
evaluation with ablations, related work, limitations). This is the
2-page compression of it for the demo track.

- **RE-XTRACT**, same system name
- $N$ concurrent **samples** of one model, never "agents" — the control
  flow is fixed, nothing plans or selects tools
- three **gates** — note this differs from the longer manuscript, which
  calls them "layers". This paper is internally consistent (zero
  occurrences of "layer" in the rendered PDF), and "gate" was chosen
  because it matches the title's *Unanimity-Gated* and because each
  stage literally admits or rejects an answer. **If both papers go out,
  align `paper/main.tex` to "gate" too** — `\TODO` markers aside, a
  reviewer who sees both will notice.

If the longer paper goes to a different venue, note that this one is
archival (AAAI proceedings, copyright transfers to AAAI), so the overlap
has to be declared.
