# Sports & Outdoors Recommender System

A large-scale recommender system built on Amazon's Sports & Outdoors review data (339K+ ratings, 73K+ items), implemented in PySpark on Databricks. Three approaches are built and compared: **collaborative filtering** (ALS), **content-based filtering** (TF-IDF + cosine similarity), and a **hybrid** blend of both.

## Highlights

- **Collaborative filtering:** Spark MLlib ALS, tuned via grid search, final RMSE **1.6817** (rank=20, regParam=0.1, seed=42)
- **Content-based filtering:** TF-IDF over title, categories, and description (vocab=500), cosine similarity to user profiles
- **Hybrid:** candidate-blending of ALS and content-based top-20 lists (α=0.5)
- **Evaluation:** RMSE, Precision@10, Recall@10, NDCG@10, Coverage, Diversity, Novelty, Serendipity@10 — compared across all three models on a 1,000-user held-out sample
- Fully distributed pipeline built to run within Databricks Free Edition serverless compute constraints

## Repo Contents

| File | Description |
|---|---|
| `FinalProject.ipynb` | Full implementation notebook — pipeline, ALS training/tuning, content-based model, hybrid model, evaluation |
| `Final_Project_Documentation.md` | Full write-up: methodology, tuning results, metric definitions, results, discussion |
| `FinalProjectPlanningDocument.ipynb` | Original planning document |

## Results

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity |
|---|---|---|---|---|---|---|
| ALS (Collaborative) | 1.6817 | 0.0000 | 0.0000 | 0.0000 | 3.96% | 0.7329 |
| Content-Based | 2.6100* | 0.0009 | 0.0028 | 0.0041 | 8.42% | 0.1364 |
| Hybrid | 2.1687* | 0.0000 | 0.0000 | 0.0000 | 4.32% | 0.7065 |

*Rescaled similarity score, not a native rating prediction.

**Key finding:** Content-Based filtering was the only model to register nonzero ranking hits and had the widest catalog coverage, but the lowest per-user diversity (a "filter bubble" effect). ALS had the best rating-prediction accuracy and the most diverse individual recommendation lists. Neither individual model dominates — see `Final_Project_Documentation.md` for full discussion.

## Method Summary

Ratings were filtered to users/items with 3+ ratings (339,076 ratings retained, well above the 10K user/item threshold). ALS was tuned via grid search on rank/regParam; a content-based model was added using TF-IDF item vectors; a hybrid model blends both signals via min-max normalized score combination. All models were evaluated on the same 1,000-user test sample.

See `Final_Project_Documentation.md` for full methodology, tuning details, and limitations.
