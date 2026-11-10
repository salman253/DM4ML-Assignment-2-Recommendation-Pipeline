import io, zipfile, urllib.request
from pathlib import Path
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def generate_data(out_dir, seed=None, **kwargs):
    # Downloads the official public MovieLens latest-small archive on first run.
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    archive=out/"ml-latest-small.zip"
    extracted=out/"ml-latest-small"
    if not (extracted/"ratings.csv").exists():
        if not archive.exists():
            request=urllib.request.Request(DATA_URL,headers={"User-Agent":"DM4ML-Assignment/1.0"})
            with urllib.request.urlopen(request,timeout=90) as response:
                archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    ratings=pd.read_csv(extracted/"ratings.csv")
    movies=pd.read_csv(extracted/"movies.csv")
    # MovieLens does not publish demographics; preserve anonymized user IDs.
    users=pd.DataFrame({"user_id":sorted(ratings.userId.unique())})
    items=movies.rename(columns={"movieId":"item_id","title":"item_title","genres":"category"}).copy()
    items["category"]=items["category"].fillna("(no genres listed)")
    interactions=ratings.rename(columns={"userId":"user_id","movieId":"item_id"}).copy()
    interactions["timestamp"]=pd.to_datetime(interactions["timestamp"],unit="s",utc=True)
    interactions["event_type"]="rating"
    interactions=interactions[["user_id","item_id","rating","timestamp","event_type"]]
    users.to_csv(out/"users.csv",index=False)
    items.to_csv(out/"items.csv",index=False)
    interactions.to_csv(out/"interactions.csv",index=False)
    return users,items,interactions
