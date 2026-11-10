import pandas as pd
from pathlib import Path

def prepare(interactions,items,out_dir,train_fraction=.8):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    x=interactions.sort_values("timestamp").copy()
    # Time-based split: latest chronological tail is held out.
    cutoff=x["timestamp"].quantile(train_fraction)
    train=x[x.timestamp<=cutoff].copy(); test=x[x.timestamp>cutoff].copy()
    train.to_csv(out/"train.csv",index=False); test.to_csv(out/"test.csv",index=False)
    user_summary=train.groupby("user_id").agg(user_activity_count=("item_id","count"),user_avg_rating=("rating","mean")).reset_index()
    item_summary=train.groupby("item_id").agg(item_popularity=("user_id","count"),item_avg_rating=("rating","mean")).reset_index()
    user_summary.to_csv(out/"user_activity_summary.csv",index=False)
    item_summary.to_csv(out/"item_popularity_summary.csv",index=False)
    return train,test,user_summary,item_summary
