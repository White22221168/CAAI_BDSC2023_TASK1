import torch
import pandas as pd
import numpy as np
import json as js
import pickle as pkl
import multiprocessing as mp

with open("triples.pickle","rb") as f:
    triples11 = pkl.load(f)
with open("sr2o.pickle","rb") as f:
    sr2o = pkl.load(f)



# 加载数据和计算权重
# weight = pd.read_csv("negsample_weight.csv")["count"]
# a = 0.1/299889
# preds = [1.0/299889]*299889
# preds = preds+a*weight
# preds = preds/preds.sum()
# userid = [x for x in range(299889)]
# for i in range (10):
#     print(i)
#     triples = triples11[i,i*100000]
#     triples_list = []
#     labels_list = []
#     for ele in triples:
#         src, rel, obj = ele
#         triples1 = [torch.LongTensor([src, rel, obj])]
#         labels = [torch.FloatTensor([1.0])]
#         while True:
#             neg_obj = np.random.choice(userid, p=preds)
#             if neg_obj not in sr2o[(src, rel)]:
#                 triples1.append(torch.LongTensor([src, rel, neg_obj]))
#                 labels.append(torch.FloatTensor([0.0]))
#             if len(triples1) > 1:
#                 break
#         triples_list.append(triples1)
#         labels_list.append(labels)
#     with open("triples_list.pickle","ab") as f:
#         pkl.dump(triples_list,f)

#     with open("labels_list.pickle","ab") as f:
#         pkl.dump(labels_list,f)


def process_task(triples, sr2o, preds, userid,k,i):
    triples_list = []
    labels_list = []
    for ele in triples:
        src, rel, obj = ele
        triples1 = [torch.LongTensor([src, rel, obj])]
        labels = [torch.FloatTensor([1.0])]
        while True:
            neg_obj = np.random.choice(userid, p=preds)
            if neg_obj not in sr2o[(src, rel)]:
                triples1.append(torch.LongTensor([src, rel, neg_obj]))
                labels.append(torch.FloatTensor([0.0]))
            if len(triples1) > 1:
                break
        triples_list.append(triples1)
        labels_list.append(labels)
    with open(f"triples_list_{k}_{i}.pickle","wb") as f:
        pkl.dump(triples_list,f)
    with open(f"labels_list_{k}_{i}.pickle","wb") as f:
        pkl.dump(labels_list,f)

if __name__ == "__main__":
    weight = pd.read_csv("negsample_weight.csv")["count"]
    a = 0.1/299889
    preds = [1.0/299889]*299889
    preds = preds+a*weight
    preds = preds/preds.sum()
    userid = [x for x in range(299889)]
    for k in range(10):
        print(f"------第{k}轮-----")
        u = k*100000
        pool = []
        for i in range(10):
            triples = triples11[u+i*10000:u+(i+1)*10000]
            pool.append(mp.Process(target=process_task, args=(triples, sr2o, preds, userid,k,i)))
        for i in range(10):
            print(i)
            pool[i].start()
        for i in range(10):
            pool[i].join()
    k=10
    print(f"------第{k}轮-----")
    u = k*100000
    pool = []
    for i in range(7):
        triples = triples11[u+i*10000:u+(i+1)*10000]
        if i==6:
            triples = triples11[u+i*10000:u+66927]
        pool.append(mp.Process(target=process_task, args=(triples, sr2o, preds, userid,k,i)))
    for i in range(7):
        print(i)
        pool[i].start()
    for i in range(7):
        pool[i].join()