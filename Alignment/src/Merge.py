# This code is to merge the datasets for Alignment training 

'''
both scripts are in same shape so ["image", "instruction","output",align_source"]

python mix.py --arm C --data_dir "C:/.../GRPO_Reasoning_PostTraining/data"
 
Arms:
    C   both sources pooled          <- your main run
    A   geo only, capped             <- ablation: caption alignment
    B   imgcode only, capped         <- ablation: code alignment
 
A and B are capped to the same size so the comparison is about the KIND of
data, not the amount. C is deliberately uncapped: it is the model you ship,
not an arm of the experiment.
'''

import argparse
import json 
import os 
import random
from collections import Counter


def load(path):
    if not os.path.exists(path):
        print("Missing path")
    
    rows = json.load(open(path, encoding="utf-8"))
    print(len(rows))
    return rows

def check_images(rows,n=200):
    sample = random.sample(rows,min(n,len(rows)))
    missing = [r["image"] for r in sample if not os.path.exists(r["image"])]
    if missing:
        print(f"\n  WARNING: {len(missing)}/{len(sample)} sampled images missing")
        print("  e.g.", missing[0])
        print("  Paths in the JSON are absolute; did you move the data folder?")
    return len(missing) 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], default="C")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--cap", type=int, default=60_000,
                    help="per-arm size for A and B. Ignored for C.")
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
 
    random.seed(args.seed)          # same split every run -> comparable results
    d = os.path.abspath(args.data_dir)

    print("Loading both the datasets")

    geo_dataset = load(os.path.join(d, r"C:\Users\Aarya-2\Documents\ADOG\MARLOW AI\GRPO_Reasoning_Postraining\GRPO_Reasoning_PostTraining\Data\Geo_170k\geo170k\data\geo_norm.json"))
    imgCode_dataset = load(os.path.join(d,r"C:\Users\Aarya-2\Documents\ADOG\MARLOW AI\GRPO_Reasoning_Postraining\GRPO_Reasoning_PostTraining\Data\imgcode_norm.json"))

    random.shuffle(geo_dataset)
    random.shuffle(imgCode_dataset)

    if args.arm == "A":
        mixed = geo_dataset[:args.cap]
    elif args.arm == "B":
        mixed = imgCode_dataset[:args.cap]
    else:
        mixed = geo_dataset + imgCode_dataset
    
    random.shuffle(mixed)

    print("check image paths of the datasets")

    check_images(mixed)

    test , train = mixed[:args.n_test] , mixed[args.n_test:]

    train_p = os.path.join(d,"align_train.json")
    test_p = os.path.join(d,"align_test_json")

    json.dump(train, open(train_p, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(test,  open(test_p,  "w", encoding="utf-8"), ensure_ascii=False)

    def summarize(name, rows):
        c = Counter(r["align_source"].split(":")[0] for r in rows)
        print(f"  {name:6s} {len(rows):>7,}   {dict(c)}")
 
    print(f"\narm {args.arm}")
    summarize("train", train)
    summarize("test",  test)
    print(f"\n  -> {train_p}")
    print(f"  -> {test_p}")

if __name__ == "__main__":
    main()