import torch
import os.path
import pandas as pd
import sys

out = sys.argv[1]

#a = pd.read_csv('batch2_metadata.tsv', sep='\t')
a = pd.read_csv('inputs/biogroup_subtypes9.csv')

ndf = []
for i, r in a.iterrows():
    #fp = f'{r.slide_id}_{r.AC_id}'
    fp = f'{r.slide_id}'
    fp2 = os.path.join(f'results/inference/preds/{fp}.pt')
    if os.path.exists(fp2):
        tl = torch.load(fp2)
        probs = tl['pred']['Y_prob'].flatten().tolist()
        class_labels = dict((v, k) for k,v in tl['metadata']['class_map'].items())
        pred = tl['pred']['Y_hat'].item()
        ndf.append({
            'slide_id': r.slide_id,
            #'AC_id': r.AC_id,
            'pred': class_labels[pred],
            'prob': probs[pred]
        })

a2 = pd.DataFrame(ndf)
a2.to_csv(out,index=False)
