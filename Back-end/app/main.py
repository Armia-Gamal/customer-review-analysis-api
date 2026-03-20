import re
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)

app = FastAPI()

# =========================================
# CPU ONLY
# =========================================

DEVICE = -1  # force CPU

# =========================================
# Sentiment Model (Stars Model)
# =========================================

sent_model_id = "nlptown/bert-base-multilingual-uncased-sentiment"

sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_id)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_id)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=sent_model,
    tokenizer=sent_tokenizer,
    device=DEVICE
)

# =========================================
# Issue Extraction Model (FLAN-T5)
# =========================================

issue_model_id = "google/flan-t5-base"

issue_tokenizer = AutoTokenizer.from_pretrained(issue_model_id)
issue_model = AutoModelForSeq2SeqLM.from_pretrained(issue_model_id)

issue_pipeline = pipeline(
    "text-generation",
    model=issue_model,
    tokenizer=issue_tokenizer,
    device=DEVICE
)

# =========================================
# Helpers
# =========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_sentiment(label):
    stars = int(label.split()[0])
    if stars <= 2:
        return "NEGATIVE"
    elif stars == 3:
        return "NEUTRAL"
    else:
        return "POSITIVE"


def severity_label(count):
    if count >= 400:
        return "Critical"
    elif count >= 200:
        return "High"
    elif count >= 80:
        return "Medium"
    else:
        return "Low"


def build_prompt(text):
    text = " ".join(str(text).split()[:200])
    return f"""
Extract the main customer issue in 2-4 words only.

Review:
{text}

Issue:
"""


def extract_issues_batch(texts, batch_size=16):
    issues = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [build_prompt(t) for t in batch]

        outputs = issue_pipeline(
            prompts,
            max_new_tokens=8,
            do_sample=False
        )

        for out in outputs:
            issue = out["generated_text"].strip().lower()
            issue = issue.split("\n")[0]
            issue = issue.replace(".", "")
            issue = " ".join(issue.split()[:4])
            issues.append(issue)

    return issues


def normalize_issue(issue, review):
    text = f"{issue} {review}".lower()

    if any(w in text for w in ["service", "support", "experience"]):
        return "poor service"

    if any(w in text for w in ["staff", "employee", "manager"]):
        return "rude staff"

    if any(w in text for w in ["wait", "delay", "slow"]):
        return "long wait time"

    if any(w in text for w in ["price", "expensive", "overpriced"]):
        return "overpriced"

    if any(w in text for w in ["quality", "dirty", "cold", "burnt"]):
        return "bad product quality"

    return "other issue"


# =========================================
# API Endpoint
# =========================================

@app.post("/analyze")
async def analyze_reviews(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(BytesIO(content))

    if "review_text" not in df.columns:
        return {"error": "CSV must contain review_text column"}

    df["review_text"] = df["review_text"].astype(str)
    df["clean_text"] = df["review_text"].apply(clean_text)

    # ======================
    # Sentiment
    # ======================

    texts = df["clean_text"].tolist()

    sentiment_results = sentiment_pipeline(
        texts,
        truncation=True,
        max_length=256
    )

    df["sentiment_raw"] = [r["label"] for r in sentiment_results]
    df["sentiment"] = df["sentiment_raw"].apply(map_sentiment)

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

    # ======================
    # Issue Extraction
    # ======================

    neg_df = df[df["sentiment"] == "NEGATIVE"].copy()

    if len(neg_df) == 0:
        return {
            "summary": {
                "total_reviews": int(len(df)),
                "sentiment_breakdown": sentiment_summary
            },
            "issue_analysis": {
                "issues": []
            }
        }

    issues_raw = extract_issues_batch(
        neg_df["review_text"].tolist()
    )

    neg_df["issue_raw"] = issues_raw

    neg_df["issue"] = neg_df.apply(
        lambda r: normalize_issue(
            r["issue_raw"],
            r["review_text"]
        ),
        axis=1
    )

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
