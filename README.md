# Sports & Outdoors Recommender System

A large-scale recommender system built on Amazon's Sports & Outdoors review data (339K+ ratings, 27K+ users, 32K+ items), implemented in PySpark on Databricks. Three approaches are built and compared: **collaborative filtering** (ALS), **content-based filtering** (TF-IDF + cosine similarity), and a **hybrid** blend of both, evaluated across seven standard recommender metrics.

## Highlights

- **Collaborative filtering:** Spark MLlib ALS, tuned via grid search (rank, regParam), final RMSE **1.2554**
- **Content-based filtering:** TF-IDF over item titles/categories, cosine similarity to rating-weighted user profiles
- **Hybrid:** candidate-blending of ALS and content-based top-N lists, normalized and re-ranked
- **Evaluation:** RMSE, Precision@10, Recall@10, NDCG@10, Coverage, Diversity, Novelty, Serendipity@10 — computed on a held-out test sample and compared across all three models
- Fully distributed pipeline (load → join → filter → index → train → evaluate) built to run within Databricks Free Edition serverless compute constraints

## Repo Contents

| File | Description |
|---|---|
| `FinalProject.ipynb` | Full implementation notebook — data pipeline, ALS training/tuning, content-based model, hybrid model, evaluation |
| `Final_Project_Documentation.md` | Full write-up: methodology, tuning results, metric definitions, results, and discussion |
| `FinalProjectPlanningDocument.ipynb` | Original planning document submitted prior to implementation |

## Quick Results

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | Novelty | Serendipity@10 |
|---|---|---|---|---|---|---|---|---|
| ALS (Collaborative) | 1.2554 | 0.00067 | 0.00133 | 0.00217 | 2.24% | 0.7438 | 16.9080 | 0.00067 |
| Content-Based | — | 0.00067 | 0.00133 | 0.00311 | 2.89% | 0.0956 | 16.7006 | 0.00067 |
| Hybrid | — | 0.00067 | 0.00167 | 0.00285 | 2.53% | 0.6790 | 16.8172 | 0.00067 |

**Key finding:** Content-Based filtering achieved the best ranking quality (NDCG@10) and highest catalog coverage, but at the cost of much lower diversity — its recommendations are more thematically narrow per user (similar items to what they already liked), while ALS spreads across more varied categories. The Hybrid model balances both, improving Recall@10 over either individual model while retaining most of ALS's diversity.

*(Full metrics table and discussion in `Final_Project_Documentation.md`, Section 5–6.)*

## Method Summary

Ratings were filtered to users/items with 3+ ratings to address sparsity while comfortably exceeding the 10K user/item scale threshold. ALS was trained on an 80/20 split and tuned across rank and regularization; a content-based model was added using TF-IDF item vectors and user content profiles; a hybrid model combines both signals via min-max normalized score blending. All models were evaluated on the same 300-user test sample using precision, recall, ranking, and beyond-accuracy metrics (coverage, diversity, novelty, serendipity).

See `Final_Project_Documentation.md` for full methodology, tuning details, and discussion of results and limitations.
