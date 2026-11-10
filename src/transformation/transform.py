import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz

def transform(train,items,out_dir):
    out=Path(out_dir)
    matrix=train.pivot_table(index="user_id",columns="item_id",values="rating",aggfunc="mean",fill_value=0)
    matrix.to_csv(out/"user_item_matrix.csv")
    save_npz(out/"user_item_matrix_sparse.npz",csr_matrix(matrix.to_numpy(dtype="float32")))
    return matrix
