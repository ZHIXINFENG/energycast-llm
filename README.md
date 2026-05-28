# EnergyCast-LLM

> Time-series electricity load forecasting powered by Temporal Fusion Transformer,
> with an LLM layer for natural language querying and result interpretation.

## Motivation

Traditional forecasting models produce numbers. This project adds a reasoning
layer on top: ask questions in plain language, get predictions back with
explanations useful for energy system operators and analysts who need both
accuracy and interpretability.

## Architecture

```
User (natural language)
        │
        ▼
  [LLM Agent Layer]          ← intent parsing, tool calling, result narration
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Forecast]  [Explain]        ← TFT model inference / feature importance
   │
   ▼
[Data Pipeline]              ← UCI Electricity Load Dataset, feature engineering
```

## Quickstart

```bash
pip install -r requirements.txt
python pipeline/prepare_data.py
python models/train.py
python interface/app.py
```

## Stack

| Component | Technology |
|---|---|
| Forecasting model | Temporal Fusion Transformer (PyTorch Forecasting) |
| LLM backend | Claude API (claude-sonnet) |
| Interface | Gradio |

## Background

Built as part of a transition from control systems / MPC into LLM-powered
forecasting applications. The forecasting layer draws on experience with
predictive control and time-series modeling; the LLM layer explores how
language models can serve as interfaces to specialized numerical models.
