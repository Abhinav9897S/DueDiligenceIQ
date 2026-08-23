# DueDiligenceIQ

A Streamlit dashboard that cross-references a company's stated ESG commitments against live public news coverage, and flags narrative divergence ("greenwashing risk") before capital is committed.

## How it works

1. **Claim retrieval** — pulls the company's official published ESG claim.
2. **Live news scrape** — fetches recent headlines + source links from Google News RSS.
3. **Sentiment analysis** — scores the claim and each headline with [FinBERT-ESG](https://huggingface.co/yiyanghkust/finbert-esg) to measure sentiment gap.
4. **Relevance scoring** — TF-IDF + cosine similarity between the claim and news coverage.
5. **Vector storage** — claim/news documents are indexed in a local ChromaDB collection.
6. **Verdict model** — a Random Forest classifier turns the sentiment discrepancy into an `INVEST` / `DO NOT INVEST` recommendation, weighed against a user-adjustable **Investor Risk Appetite** threshold (Conservative / Balanced / Aggressive).

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

First run will download the FinBERT-ESG model weights, so it may take a minute.

## Authors

- Abhinav Singh
- Aryaman Gora
- Aarav Singh
