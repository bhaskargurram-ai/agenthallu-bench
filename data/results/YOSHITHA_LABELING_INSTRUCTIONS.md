# Second-Annotator Labeling — Instructions (Yoshitha)

**Goal:** provide an independent second set of labels on the *same 100 traces*
the first author labeled, so we can report a real inter-annotator agreement
(Cohen's κ) for the judge-reliability study. You are acknowledged (not a
co-author).

**File to fill:** `human_labeling_sheet_Yoshitha_BLANK.csv` (100 rows, identical
`session_id`s to the first author's sheet).

**Your task:** for each row, read `query`, `ground_truth` (the reference/correct
answer), and `final_answer` (what the agent produced). Put your judgment in the
`YOUR_LABEL` column:

- `YES` — the agent's final answer is substantively correct (matches the core
  fact in `ground_truth`). Accept minor wording differences, different units,
  or extra explanation as long as the core fact is right.
- `NO`  — the agent's final answer is wrong, missing the core fact, or is a
  non-answer (e.g. "Max steps reached.").

**Do not** look at the first author's labels while labeling (independence).
Label all 100 if possible; a minimum of 50 gives a usable κ.

When done, save as `human_labeling_sheet_Yoshitha_labeled.csv` in this folder.
I will convert it to `human_labels_yoshitha.csv` and compute κ automatically.
