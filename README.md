# MovieLens End-to-End Recommendation Data Pipeline

## Dataset choice and source
This project uses the **MovieLens latest-small** public educational dataset published by GroupLens Research, University of Minnesota:
- Dataset page: https://grouplens.org/datasets/movielens/
- Direct archive: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Dataset documentation: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
- Required academic citation: Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4). https://doi.org/10.1145/2827872

The latest-small release contains 100,836 ratings, 9,742 movies, 610 anonymized users and 3,683 tag applications (the pipeline uses ratings and movie metadata). The source dataset documentation states that each included user rated at least 20 movies. User demographics are not supplied, so `users.csv` correctly contains anonymized user IDs only. Movie genres are pipe-separated. Rating timestamps are Unix seconds and are converted to UTC timestamps.

**Use terms:** GroupLens asks users to review the README/terms before use and notes that permission is required for commercial use. This is an educational/non-commercial assignment. The repository downloads the archive from GroupLens at first execution rather than redistributing a copy of the dataset.

## Environment and execution
Python 3.10 or 3.11 recommended.
```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```
The first run needs internet access to download the official MovieLens ZIP. Subsequent runs reuse `data/raw/ml-latest-small/`. Core pipeline uses pandas, NumPy, scikit-learn, SciPy and joblib. MLflow is optional and is used when installed. Airflow is optional; install it separately with the official constraints file matching your Python version and OS.

## Pipeline stages
1. **Ingestion:** downloads/extracts MovieLens and writes `users.csv`, `items.csv`, `interactions.csv`. `users.csv` has anonymized IDs only because no demographics are provided.
2. **Validation:** checks required fields, duplicate user-item pairs, valid half-star ratings in [0.5, 5.0], timestamps and referential integrity. Invalid rows are separated to `data/errors/validation_errors.csv`; counts go to `reports/validation_report.json`.
3. **Preparation:** sorts by UTC timestamp and creates a chronological 80/20 holdout split. It produces train/test CSVs, user activity and item popularity summaries.
4. **Transformation:** creates a user-item rating matrix in CSV and sparse NPZ formats.
5. **Feature management:** user activity count, mean rating, preferred genre and recency; item popularity, mean rating and genres; interaction rating, timestamp and event type.
6. **Training:** popularity baseline and personalized item-item collaborative filtering using cosine similarity. Sparse similarity is used to avoid a dense movie-by-movie similarity matrix. Models are serialized with joblib; experiment metadata is written locally and MLflow logging is attempted if available.
7. **Evaluation:** Precision@5, Recall@5, NDCG@5, catalog Coverage and genre Diversity. RMSE/MAE are not applicable because these are top-K ranking models, not rating predictors.
8. **Orchestration:** `src/pipeline.py` runs all stages in dependency order. `dags/recommendation_pipeline_dag.py` wraps the same command in an Airflow DAG.

## Outputs
- `data/raw/`: downloaded archive and extracted MovieLens CSV files plus normalized tables
- `data/errors/`: quarantined invalid records
- `data/processed/`: chronological train/test sets, summaries and interaction matrices
- `data/features/`: user, item and interaction feature tables
- `models/`: popularity and item-CF joblib artifacts
- `reports/`: validation report, evaluation table, run summary and feature catalog
- `mlruns/`: experiment metadata and MLflow artifacts when enabled

## Feature catalog
| Entity | Feature | Definition |
|---|---|---|
| User | user_activity_count | Number of training ratings |
| User | user_avg_rating | Mean training rating |
| User | preferred_category | Most frequently rated genre in training |
| User | days_since_last_interaction | Days from latest training timestamp to user's last rating |
| Item | item_popularity | Number of training ratings |
| Item | item_avg_rating | Mean training rating |
| Item | category | Pipe-separated MovieLens genres |
| Item | item_title | Movie title |
| Interaction | rating | Explicit MovieLens rating, 0.5–5.0 |
| Interaction | timestamp | UTC rating timestamp |
| Interaction | event_type | Constant `rating`, derived from source ratings |

## Design decisions and limitations
The chronological holdout reduces leakage from future ratings into training. The baseline ranks by rating count. Item-item CF computes cosine similarity over sparse user-item ratings. The diversity metric is defined as the proportion of unique genre labels across the recommended list (multi-genre movies contribute each listed genre). The dataset has anonymized user IDs but no user demographics. The latest-small dataset is intended for education/development, not stable benchmark claims. Cold-start behavior is limited, and production use would need online serving, monitoring, privacy controls and retraining policy.

## Reproducibility and submission
Configuration is in `configs/config.json`. Run `python -m src.pipeline` from the repository root. The script prints metrics and writes them to `reports/evaluation_results.csv`. Cite the MovieLens dataset and paper in reports and presentations.

The item-item collaborative filtering model is generated during pipeline execution and is not committed because of its large file size. The popularity model is included as a sample trained artifact.

## Report and execution evidence

The final formatted assignment report is available at `reports/final_report.pdf`.

The report documents the MovieLens dataset, pipeline implementation, validation results, model evaluation, Airflow orchestration and execution evidence.

Airflow DAG and successful task execution screenshots are available in the `screenshots/` directory.

The script `reports/generate_report.py` can be used to regenerate the report after running the pipeline.
