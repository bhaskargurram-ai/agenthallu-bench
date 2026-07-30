"""AgentProp-Bench leaderboard — Hugging Face Space (Gradio).

Renders leaderboard.csv. New models are added by running
`agentprop eval --model X` locally and opening a PR that appends a row to
leaderboard.csv (see the "Submit a model" tab).
"""
import os

import gradio as gr
import pandas as pd

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "leaderboard.csv")

INTRO = """
# 🧭 AgentProp-Bench Leaderboard

Tool-using LLM agents scored on what answer-only benchmarks miss:
**error propagation**, **tool-call fabrication**, and the **judge gap**.

- **Propagation (EPS)** — how strongly a single injected wrong tool parameter
  propagates toward a wrong final answer. Higher = less robust.
- **Fabrication %** — traces where the model emits a tool *result* it never obtained.
- **Acc (validated judge)** vs **Acc (substring judge)** — the same runs scored by a
  validated 3-LLM ensemble vs. naive substring matching. The gap is why judge
  choice, not the model, often decides the leaderboard (substring κ=0.049 vs
  ensemble κ=0.432 against 100 human labels).

📄 [Paper (arXiv:2604.16706)](https://arxiv.org/abs/2604.16706) ·
💻 [Code](https://github.com/bhaskargurram-ai/agenthallu-bench) ·
🤗 Dataset: `AgentProp-Bench`
"""

SUBMIT = """
## Submit a model

```bash
pip install agentprop-bench
export OPENAI_API_KEY=...        # or GOOGLE_API_KEY / OPENROUTER_API_KEY / VLLM_BASE_URL
agentprop eval --model your-model-id --out my_row.csv
```

Then open a PR on the GitHub repo appending your row to `leaderboard/leaderboard.csv`,
or post the row in this Space's Community tab. Include your run config (task file,
n, judge) so the row is reproducible.
"""


def load():
    df = pd.read_csv(CSV)
    df = df.rename(columns={
        "model": "Model", "kind": "Type", "n": "N",
        "propagation_eps": "Propagation (EPS ↓)",
        "reached_output_rate": "Reached output ↓",
        "fabrication_pct": "Fabrication % ↓",
        "tool_call_pct": "Tool-call %",
        "acc_validated_judge": "Acc (validated) ↑",
        "acc_substring_judge": "Acc (substring)",
    })
    return df


def filtered(kind):
    df = load()
    if kind != "all":
        df = df[df["Type"] == kind]
    return df


with gr.Blocks(title="AgentProp-Bench Leaderboard") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Leaderboard"):
        radio = gr.Radio(["all", "proprietary", "open-weight"], value="all", label="Filter")
        table = gr.Dataframe(value=load(), interactive=False, wrap=True)
        radio.change(filtered, inputs=radio, outputs=table)
    with gr.Tab("Submit a model"):
        gr.Markdown(SUBMIT)

if __name__ == "__main__":
    demo.launch()
