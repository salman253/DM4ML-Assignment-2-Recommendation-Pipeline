import shlex

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

PROJECT_ROOT = r"""/mnt/d/SALMAN/M.Tech/SEM 2 July 2026 - Dec 2026/Data Management for ML (S1-26_AIMLZG529)/DM4ML Assignment 2/recommendation-pipeline-public-dataset/recommendation-pipeline"""
PYTHON = "/home/salman/recommendation-venv/bin/python"

with DAG(
    dag_id="recommendation_data_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["DM4ML", "MovieLens"],
) as dag:

    run_pipeline = BashOperator(
        task_id="run_recommendation_pipeline",
        bash_command=f"{shlex.quote(PYTHON)} -m src.pipeline",
        cwd=PROJECT_ROOT,
    )
