import numpy as np, pandas as pd, joblib, json
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

class PopularityRecommender:
    def fit(self,train):
        self.scores=train.groupby("item_id").rating.count().sort_values(ascending=False)
        self.items=list(self.scores.index); return self
    def recommend(self,user_id,seen,k=5):
        return [i for i in self.items if i not in seen][:k]

class ItemCFRecommender:
    def fit(self,train):
        self.matrix=train.pivot_table(index="user_id",columns="item_id",values="rating",aggfunc="mean",fill_value=0)
        self.user_ids=list(self.matrix.index); self.item_ids=list(self.matrix.columns)
        sparse=csr_matrix(self.matrix.to_numpy(dtype=np.float32))
        self.sim=cosine_similarity(sparse.T,dense_output=False).tocsr()
        self.item_index={v:i for i,v in enumerate(self.item_ids)}
        return self
    def recommend(self,user_id,seen,k=5):
        if user_id not in self.matrix.index: return []
        row=self.matrix.loc[user_id].to_numpy(dtype=np.float32)
        scores=np.asarray(row @ self.sim).ravel()
        if seen:
            for item in seen:
                idx=self.item_index.get(item)
                if idx is not None: scores[idx]=-np.inf
        ranked=np.argsort(scores)[::-1]
        return [self.item_ids[j] for j in ranked if np.isfinite(scores[j])][:k]

def train_models(train,model_dir,tracking_dir):
    p=PopularityRecommender().fit(train); cf=ItemCFRecommender().fit(train)
    Path(model_dir).mkdir(parents=True,exist_ok=True); Path(tracking_dir).mkdir(parents=True,exist_ok=True)
    joblib.dump(p,Path(model_dir)/"popularity.joblib"); joblib.dump(cf,Path(model_dir)/"item_cf.joblib")
    run={"experiment":"movielens_recommendation_pipeline","dataset":"MovieLens latest-small",
         "models":[{"name":"popularity","parameters":{"strategy":"interaction_count"}},
                   {"name":"item_item_cf","parameters":{"similarity":"cosine","rating_signal":"explicit_rating"}}]}
    (Path(tracking_dir)/"experiment.json").write_text(json.dumps(run,indent=2))
    try:
        import mlflow
        mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
        mlflow.set_experiment("movielens-recommendation-pipeline")
        with mlflow.start_run(run_name="popularity-and-item-cf"):
            mlflow.log_param("similarity","cosine"); mlflow.log_param("top_k",5)
            mlflow.log_metric("training_interactions",len(train))
            mlflow.log_artifact(str(Path(model_dir)/"popularity.joblib"))
            mlflow.log_artifact(str(Path(model_dir)/"item_cf.joblib"))
    except Exception as e:
        (Path(tracking_dir)/"mlflow_note.txt").write_text("MLflow optional; experiment metadata retained. "+str(e))
    return p,cf


def log_evaluation_to_mlflow(results, tracking_dir, model_dir, report_dir):
    """
    Log evaluation metrics for each recommender as separate MLflow runs.
    """
    from pathlib import Path
    import mlflow

    tracking_path = Path(tracking_dir).resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)

    database_path = tracking_path / "mlflow.db"
    mlflow.set_tracking_uri(
    	f"sqlite:///{database_path.as_posix()}"
    )

    mlflow.set_experiment("movielens-recommendation-pipeline")

    metric_columns = [
        "Precision@5",
        "Recall@5",
        "NDCG@5",
        "Coverage",
        "Diversity",
    ]

    for _, row in results.iterrows():
        model_name = str(row["Model"])

        with mlflow.start_run(run_name=f"{model_name}-evaluation"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("k", 5)
            mlflow.log_param("dataset", "MovieLens latest-small")

            for metric in metric_columns:
                mlflow.log_metric(metric.replace("@", "_at_"), float(row[metric]))

            model_file = (
                "popularity.joblib"
                if model_name == "Popularity"
                else "item_cf.joblib"
            )

            artifact_path = Path(model_dir) / model_file
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="models")

            evaluation_file = Path(report_dir) / "evaluation_results.csv"
            if evaluation_file.exists():
                mlflow.log_artifact(str(evaluation_file), artifact_path="evaluation")

    print("MLflow evaluation metrics logged successfully.")
