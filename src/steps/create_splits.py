import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from lib.etc import read_metadata

def main(cfg):
    target_col = cfg.common.target_label
    tbl = read_metadata(cfg.splits.input, check_cols=['case_id',target_col])

    m2 = tbl[['case_id',target_col]].drop_duplicates().dropna()
    if not len(m2.case_id.unique()) == len(m2):
        raise Exception("there is duplicate labels per case_id on metadata")

    gss = StratifiedShuffleSplit(n_splits=cfg.splits.n_folds, test_size=cfg.splits.valid_perc)

    z = list(gss.split(X=m2.case_id, y=m2[target_col]))

    dd = []
    for fold_idx, fold in enumerate(z):
        for is_valid, items in enumerate(fold):
            for item in items:
                rec = m2.iloc[item].to_dict()
                rec['fold'] = fold_idx
                rec['is_valid'] = is_valid
                dd.append(rec)

    m3 = pd.DataFrame.from_dict(dd)
    m = tbl.merge(m3, on=['case_id',target_col], how='inner')
    m.to_csv(cfg.splits.output, index=False)
