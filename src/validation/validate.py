import pandas as pd
from pathlib import Path

def validate(users,items,interactions,error_dir):
    x=interactions.copy()
    x["rating_num"]=pd.to_numeric(x["rating"],errors="coerce")
    x["parsed_timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce",utc=True)
    checks={
      "missing_required":x[["user_id","item_id","rating","timestamp"]].isna().any(axis=1),
      "duplicate_user_item":x.duplicated(["user_id","item_id"],keep="first"),
      "invalid_rating":~x["rating_num"].between(0.5,5.0) | ((x["rating_num"]*2)%1 != 0),
      "invalid_timestamp":x["parsed_timestamp"].isna(),
      "unknown_user":~x["user_id"].isin(users.user_id),
      "unknown_item":~x["item_id"].isin(items.item_id)
    }
    errors=[]; invalid=pd.Series(False,index=x.index)
    for reason,mask in checks.items():
        mask=mask.fillna(True); invalid|=mask
        if mask.any():
            part=x.loc[mask].copy(); part["error_reason"]=reason; errors.append(part)
    clean=x.loc[~invalid].copy()
    clean["rating"]=clean["rating_num"].astype(float); clean["timestamp"]=clean["parsed_timestamp"]
    clean=clean.drop(columns=["rating_num","parsed_timestamp"])
    err=pd.concat(errors,ignore_index=True).drop_duplicates() if errors else pd.DataFrame(columns=list(x.columns)+["error_reason"])
    Path(error_dir).mkdir(parents=True,exist_ok=True); err.to_csv(Path(error_dir)/"validation_errors.csv",index=False)
    report={k:int(v.sum()) for k,v in checks.items()}
    report.update({"input_rows":len(x),"valid_rows":len(clean),"invalid_rows":int(invalid.sum())})
    return clean,report
