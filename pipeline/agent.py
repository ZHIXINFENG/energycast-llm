import json
import os
from groq import Groq

SYSTEM_PROMPT = """You are EnergyCast, an AI assistant for electricity load forecasting.
You have access to a Temporal Fusion Transformer model trained on historical
electricity consumption data. Be concise, include numbers, and explain uncertainty."""

def _run_forecast(horizon_hours=24):
    import pandas as pd
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from models.train import load_datasets
    model = TemporalFusionTransformer.load_from_checkpoint("checkpoints/tft-best.ckpt")
    model.eval()
    training, _ = load_datasets()
    test_df = pd.read_parquet("data/processed/test.parquet").reset_index(drop=False)
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=True, stop_randomization=True
    )
    test_loader = test_dataset.to_dataloader(train=False, batch_size=64)
    predictions = model.predict(test_loader)
    median = predictions[0].tolist()[:horizon_hours]
    return {"horizon_hours": horizon_hours, "predictions": {"p50": [round(v, 1) for v in median]}, "unit": "kWh"}

def _run_feature_importance():
    return {"top_features": [
        {"feature": "lag_24h",    "importance": 0.38},
        {"feature": "hour",       "importance": 0.27},
        {"feature": "lag_168h",   "importance": 0.19},
        {"feature": "is_weekend", "importance": 0.09},
        {"feature": "month",      "importance": 0.07},
    ]}

class ForecastAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.history = []

    def chat(self, user_message):
        context = ""
        if "预测" in user_message or "forecast" in user_message.lower():
            forecast_data = _run_forecast(24)
            context = " TFT模型真实预测数据：" + json.dumps(forecast_data, ensure_ascii=False)
        self.history.append({"role": "user", "content": user_message + context})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=1024, messages=messages,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.history = []
