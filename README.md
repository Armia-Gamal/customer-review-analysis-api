# LLM Customer Review Analysis System

## Overview

This project is a complete AI-powered system for analyzing customer reviews using Large Language Models and deep learning techniques.

It provides:

* Sentiment analysis (Positive / Neutral / Negative)
* Issue extraction from negative reviews
* Issue categorization and normalization
* Severity analysis based on frequency
* Full API deployment using FastAPI

The system is designed to handle real-world datasets and scale to production environments.

---

## Project Structure

```
Back-end
│   Dockerfile
│   requirements.txt
│
└── app
    └── main.py

Front-end
│   app.py

Notbooks
│   llm-sentiment-issue-extraction.ipynb
```

---

## System Pipeline

The system follows a structured pipeline:

1. Load data (CSV / Excel / TXT)
2. Clean and preprocess text
3. Perform sentiment analysis
4. Filter negative reviews
5. Extract issues using LLM (FLAN-T5)
6. Normalize issue categories
7. Compute issue frequency
8. Assign severity levels
9. Return structured JSON output

---

## Features

* Large-scale sentiment analysis using transformer models
* Issue extraction using generative LLM (FLAN-T5)
* Batch processing for performance optimization
* GPU support for faster inference
* Automatic issue categorization
* Severity classification based on data distribution
* REST API for integration with frontend or external systems

---

## Models Used

### Sentiment Analysis

* Model: nlptown/bert-base-multilingual-uncased-sentiment
* Task: Text Classification
* Output:

  * Sentiment label (1–5 stars → mapped to Positive / Neutral / Negative)

---

### Issue Extraction

* Model: google/flan-t5-base (or flan-t5-small for lightweight deployment)
* Task: Text-to-Text Generation
* Output:

  * Short issue phrase (2–4 words)

---

## Data Processing

### Text Cleaning

* Lowercasing
* Removing URLs
* Removing special characters
* Removing stopwords
* Normalizing whitespace

---

### Issue Normalization

Issues are mapped into standardized categories:

* poor service
* rude staff
* long wait time
* overpriced
* bad product quality
* bad management
* other issue

---

## Severity Calculation

Severity is determined based on frequency:

* Critical: very high frequency
* High: high frequency
* Medium: moderate frequency
* Low: low frequency

---

## Output Format

The system returns structured JSON:

```
{
  "summary": {
    "total_reviews": int,
    "sentiment_breakdown": [
      {
        "sentiment": "POSITIVE | NEGATIVE | NEUTRAL",
        "count": int,
        "percentage": float
      }
    ]
  },
  "issue_analysis": {
    "issues": [
      {
        "issue": string,
        "count": int,
        "severity": string
      }
    ]
  }
}
```

---

## Backend (FastAPI)

The backend provides:

* `/health` → check API status
* `/analyze` → upload CSV file and get analysis

### Requirements

* Python 3.10+
* FastAPI
* Transformers
* PyTorch
* Pandas
* NLTK

---

## Running the Backend

### Install dependencies

```
pip install -r requirements.txt
```

### Run server

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker Setup

### Build image

```
docker build -t review-analysis-api .
```

### Run container

```
docker run -p 8000:8000 review-analysis-api
```

---

## Notebook

The notebook includes:

* Data upload interface
* Full preprocessing pipeline
* Model loading on GPU
* Batch inference
* Visualization (charts)
* JSON export

---

## Frontend

A simple frontend interface is provided to interact with the API.

---

## Performance Considerations

* Batch processing is used for scalability
* GPU acceleration improves inference speed
* Lightweight models can be used for deployment environments
* CPU fallback supported

---

## Future Improvements

* Real-time streaming analysis
* Dashboard visualization
* Multi-language support
* Advanced topic modeling
* Integration with business intelligence tools
* Deployment on cloud platforms

---

## Summary

This project delivers a production-ready AI system for customer review analysis using modern NLP and LLM techniques.

It combines classification, generation, and data analytics into a single scalable pipeline suitable for real-world applications.
