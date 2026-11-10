import numpy as np, pandas as pd
from pathlib import Path

def evaluate(model,train,test,items,k=5):
    truth=test.groupby("user_id").item_id.apply(set).to_dict()
    seen=train.groupby("user_id").item_id.apply(set).to_dict()
    catalog=set(items.item_id); precisions=[]; recalls=[]; ndcgs=[]; recommended=set(); intra=[]
    category=items.set_index("item_id").category.fillna("(no genres listed)").str.split("|").to_dict()
    for user,actual in truth.items():
        rec=model.recommend(user,seen.get(user,set()),k); recommended.update(rec)
        hits=[1 if x in actual else 0 for x in rec]
        precisions.append(sum(hits)/k); recalls.append(sum(hits)/len(actual) if actual else 0)
        dcg=sum(hit/np.log2(idx+2) for idx,hit in enumerate(hits))
        ideal=sum(1/np.log2(idx+2) for idx in range(min(len(actual),k)))
        ndcgs.append(dcg/ideal if ideal else 0)
        cats=[genre for x in rec if x in category for genre in category[x]]
        if len(cats)>1: intra.append(len(set(cats))/len(cats))
    return {"Precision@5":float(np.mean(precisions)),"Recall@5":float(np.mean(recalls)),
      "NDCG@5":float(np.mean(ndcgs)),"Coverage":len(recommended)/len(catalog),
      "Diversity":float(np.mean(intra)) if intra else 0.0}

def evaluate_models(pop,cf,train,test,items,out_dir):
    rows=[]
    for name,model in [("Popularity",pop),("Item-item collaborative filtering",cf)]:
        rows.append({"Model":name,**evaluate(model,train,test,items,5)})
    df=pd.DataFrame(rows); Path(out_dir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(out_dir)/"evaluation_results.csv",index=False)
    return df
