import pandas as pd
from pathlib import Path

def build_features(train,users,items,out_dir):
    u=train.groupby("user_id").agg(user_activity_count=("item_id","count"),user_avg_rating=("rating","mean"),
        last_interaction=("timestamp","max")).reset_index()
    i=train.groupby("item_id").agg(item_popularity=("user_id","count"),item_avg_rating=("rating","mean")).reset_index()
    i=i.merge(items[["item_id","item_title","category"]],on="item_id",how="left")
    # MovieLens genres are pipe-separated; derive each user's most frequently rated genre.
    genre_rows=train.merge(items[["item_id","category"]],on="item_id",how="left")
    genre_rows["category"]=genre_rows["category"].fillna("(no genres listed)").str.split("|")
    genre_rows=genre_rows.explode("category")
    pref=genre_rows.groupby(["user_id","category"]).size().reset_index(name="n")
    pref=pref.sort_values(["user_id","n","category"],ascending=[True,False,True]).drop_duplicates("user_id")
    u=u.merge(pref[["user_id","category"]].rename(columns={"category":"preferred_category"}),on="user_id",how="left")
    u["days_since_last_interaction"]=(train.timestamp.max()-u.last_interaction).dt.total_seconds()/86400
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    u.to_csv(out/"user_features.csv",index=False); i.to_csv(out/"item_features.csv",index=False)
    train.to_csv(out/"interaction_features.csv",index=False)
    return u,i
