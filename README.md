# Movie Recommendation Systems: Method Comparison

An applied machine learning project that implements and compares multiple
recommendation system approaches, focusing on model behavior, interpretability,
and evaluation rather than production deployment.

---

## 30-Second Quick View

This project explores three widely used recommendation system paradigms:

- Content-Based Filtering
- Item-Based Collaborative Filtering
- Neural Collaborative Filtering (NCF)

Each method is implemented separately and evaluated using the same dataset and
metric (MAE), enabling a structured comparison of their assumptions, strengths,
and limitations.

The goal of this project is to demonstrate practical understanding of
recommender system design and evaluation in aN applied context.

---

## Dataset


This project uses the movies dataset available on Kaggle:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Particularly used files include:
- `movies_metadata.csv`
- `credits.csv`
- `ratings.csv`
- `keywords.csv`
- `links.csv`
- `links_small.csv`
- `ratings_small.csv`

The dataset is available under Kaggle’s terms and is **not included in this repository**.

---

## Methods

### 1. Content-Based Filtering

Implemented using item metadata (e.g., genres, keywords, cast, director):

- Text features are vectorized using `CountVectorizer`
- Item similarity is computed via cosine similarity
- Predictions are generated based on similarity-weighted user preferences

This approach emphasizes **feature engineering and interpretability**.

---

### 2. Item-Based Collaborative Filtering

A memory-based collaborative filtering approach:

- User–item rating matrix construction
- Mean-centered normalization
- Item–item similarity via cosine similarity
- Rating prediction using weighted averages

This method highlights **similarity-based modeling and rating normalization**.

---

### 3. Neural Collaborative Filtering (NCF)

A model-based approach implemented in a Jupyter Notebook:

- User and item embeddings
- Neural network–based interaction modeling
- End-to-end learning of latent representations

This method demonstrates a **deep learning alternative** to similarity-based
recommender systems.

---

## Evaluation Strategy

All models are evaluated using:

- **Mean Absolute Error (MAE)** on held-out test data
- Multiple random seeds (where applicable) to reduce evaluation variance

This ensures a fair and consistent comparison across methods.

---

## Results Summary

The three approaches exhibit different trade-offs:

- Content-based filtering offers strong interpretability but limited personalization
- Item-based collaborative filtering captures collective behavior patterns
- Neural collaborative filtering provides modeling flexibility at the cost of
  interpretability and complexity

The comparison highlights how model choice depends on data availability,
interpretability requirements, and system goals.

---

## Code Structure

```text
recommender-systems-comparison/
├── content_based.py
├── item_based.py
├── NCF.ipynb
└── README.md
```

## Project Scope
This repository focuses on:

- Algorithmic understanding of recommender systems
- Method comparison and evaluation
- Applied machine learning reasoning




