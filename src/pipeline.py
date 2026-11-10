import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ingestion.generate_data import generate_data
from src.validation.validate import validate
from src.preparation.prepare import prepare
from src.transformation.transform import transform
from src.features.build_features import build_features
from src.training.train import train_models
from src.evaluation.evaluate import evaluate_models

def run_pipeline():
    cfg=json.loads((ROOT/"configs/config.json").read_text()); seed=cfg["random_seed"]
    raw=ROOT/"data/raw"; processed=ROOT/"data/processed"; features=ROOT/"data/features"
    users,items,interactions=generate_data(raw)
    clean,validation_report=validate(users,items,interactions,ROOT/"data/errors")
    (ROOT/"reports/validation_report.json").write_text(json.dumps(validation_report,indent=2))
    train,test,user_summary,item_summary=prepare(clean,items,processed,cfg["train_fraction"])
    transform(train,items,processed)
    build_features(train,users,items,features)
    pop,cf=train_models(train,ROOT/"models",ROOT/"mlruns")
    results=evaluate_models(pop,cf,train,test,items,ROOT/"reports")
    summary={"status":"SUCCESS","raw_interactions":len(interactions),"valid_interactions":len(clean),
      "train_rows":len(train),"test_rows":len(test),"models":2,"dataset":"MovieLens latest-small","users":int(users.user_id.nunique()),"items":int(items.item_id.nunique())}
    (ROOT/"reports/run_summary.json").write_text(json.dumps(summary,indent=2))
    from reports.generate_report import generate_report
    generate_report(ROOT)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(json.dumps(summary,indent=2))
    print("\nMODEL EVALUATION")
    print(results.to_string(index=False))
    print("\nValidation:",validation_report)
    return summary

if __name__=="__main__": run_pipeline()
