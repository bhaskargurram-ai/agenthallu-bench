---
title: AgentProp-Bench Leaderboard
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# AgentProp-Bench Leaderboard

A Hugging Face Space that renders the AgentProp-Bench leaderboard for tool-using
LLM agents — error propagation, tool-call fabrication, and the judge gap.

- Paper: https://arxiv.org/abs/2604.16706
- Code: https://github.com/bhaskargurram-ai/agenthallu-bench

## Deploy

```bash
huggingface-cli repo create AgentProp-Bench-Leaderboard --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<user>/AgentProp-Bench-Leaderboard
cp leaderboard/app.py leaderboard/leaderboard.csv leaderboard/requirements.txt <clone>/
cd <clone> && git add -A && git commit -m "seed leaderboard (13 models)" && git push
```

`leaderboard.csv` is regenerated from the released results with
`python analysis/build_leaderboard.py`.
