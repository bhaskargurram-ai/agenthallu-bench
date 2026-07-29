# IEEE Access — Submission Package & Checklist

**Manuscript:** Evaluating Tool-Using Language Agents: Judge Reliability,
Propagation Cascades, and Runtime Mitigation in AgentProp-Bench
**Author:** Bhaskar Gurram (Zasti Inc.) · **Type:** Research Article

## Files in this package (`AgentProp_IEEE_submission.zip`)
| File | Purpose | Upload as |
|------|---------|-----------|
| `main.pdf` | Compiled manuscript (12 pp) | Main manuscript (PDF) |
| `main.tex` + `main.bbl` + `refs.bib` | LaTeX source | Source files |
| `ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, fonts | Template | Source files |
| `tables/`, `figures/` | Assets | Source files |
| `supplementary.pdf` / `supplementary.tex` | Supplementary material (6 pp) | Supplementary (reviewed) |
| `cover_letter.pdf` / `cover_letter.tex` | Cover letter | Cover letter / comments |

Rebuild everything with: `bash analysis/finalize_openmodels.sh` (from repo root).

## Pre-submission checklist (maps to the IEEE Access checklist)
- [x] IEEE Access double-column template (`ieeeaccess.cls`)
- [x] LaTeX source **and** PDF; content matches; < 40 MB (bundle ≈ 2.5 MB)
- [x] ≤ 20 pages (12 pp)
- [x] Abstract 150–250 words (233), self-contained
- [x] 3–10 keywords (8 index terms)
- [x] Acronyms defined on first use (LLM, IR, RAG, EPS, …)
- [x] Author bio after references (`IEEEbiographynophoto`)
- [x] References resolve; none retracted; recent (2024–2026) IR/agent work cited
- [x] Not under review elsewhere (IP&M submission withdrawn)
- [x] Code/data public and cited (github.com/bhaskargurram-ai/agenthallu-bench)

## Actions to complete in the submission portal (only you can do these)
1. **ORCID** — the submitting author's ORCID must be linked and public.
2. **Manuscript type** — select **Research Article**.
3. **Keywords** — paste the 8 index terms.
4. **Author list** — confirm the portal extracts "Bhaskar Gurram" from the source.
5. **Suggested/opposed reviewers** — optional.
6. **Run Paperpal Preflight** (grammar) if prompted — the manuscript is already
   copy-edited, but IEEE offers this as a courtesy check.

## Notes
- The manuscript type could alternatively be **Applied Research** (practical
  systems + quantitative validation); Research Article is the safe default.
- Optional strengthening: a second annotator's labels on the prepared
  `data/results/human_labeling_sheet_Yoshitha_BLANK.csv` would upgrade the
  inter-annotator note to a full Cohen's κ.
