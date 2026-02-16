import os
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import re

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_URL = f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}"


# ==============================
# Helpers
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def call_hf(text):
    response = requests.post(
        SENTIMENT_URL,
        headers=HEADERS,
        json={"inputs": text[:512]}
    )
    result = response.json()

    if isinstance(result, list):
        return result[0]["label"]
    return "NEUTRAL"


def normalize_issue(review):
    review = review.lower()

    if any(w in review for w in ["service", "support", "experience"]):
        return "poor service"

    if any(w in review for w in ["staff", "employee", "manager"]):
        return "rude staff"

    if any(w in review for w in ["wait", "waiting", "delay"]):
        return "long wait time"

    if any(w in review for w in ["price", "expensive", "overpriced"]):
        return "overpriced"

    if any(w in review for w in ["quality", "dirty", "cold", "burnt"]):
        return "bad product quality"

    return "other issue"


def severity_label(count):
    if count >= 400:
        return "Critical"
    elif count >= 200:
        return "High"
    elif count >= 80:
        return "Medium"
    return "Low"


# ==============================
# API
# ==============================

@app.post("/analyze")
async def analyze_reviews(file: UploadFile = File(...)):

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))
    except:
        df = pd.read_csv(BytesIO(content), encoding="latin-1")

    if "review_text" not in df.columns:
        return {"error": "CSV must contain review_text column"}

    df["review_text"] = df["review_text"].astype(str)
    df["clean_text"] = df["review_text"].apply(clean_text)

    # =====================
    # Sentiment (HF API)
    # =====================

    df["sentiment"] = df["clean_text"].apply(call_hf)

    sentiment_counts = df["sentiment"].value_counts()
    sentiment_percentages = (
        sentiment_counts / sentiment_counts.sum() * 100
    ).round(2)

    sentiment_summary = [
        {
            "sentiment": s,
            "count": int(sentiment_counts[s]),
            "percentage": float(sentiment_percentages[s])
        }
        for s in sentiment_counts.index
    ]

    # =====================
    # Issues
    # =====================

    neg_df = df[df["sentiment"] == "NEGATIVE"].copy()
    neg_df["issue"] = neg_df["review_text"].apply(normalize_issue)

    issue_counts = (
        neg_df["issue"]
        .value_counts()
        .reset_index()
    )

    issue_counts.columns = ["issue", "count"]
    issue_counts["severity"] = issue_counts["count"].apply(severity_label)

    return {
        "summary": {
            "total_reviews": int(len(df)),
            "sentiment_breakdown": sentiment_summary
        },
        "issue_analysis": {
            "issues": issue_counts.to_dict(orient="records")
        }
    }
