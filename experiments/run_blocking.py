import pandas as pd
import numpy as np
import os
import json
import random
import re
import unicodedata
import time

def normalize_text(text):
    if not isinstance(text, str) or not text: return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_prefix(text, n=4):
    if not text: return ""
    return text[:n]

def extract_token(text):
    if not text: return ""
    parts = text.split()
    return parts[0] if parts else ""

data_dir = "/home/balaji/Projects/entity-resolution/student_resource/dataset"
gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

sampled_s1_ids = []
gt_s2_matches = {}
gt_s3_matches = {}

with open(gt_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        s1, matched = line.rstrip('\n').split('\t')
        matches = [x.strip() for x in matched.split(',') if x.strip()]
        if matches:
            sampled_s1_ids.append(s1)
            gt_s2_matches[s1] = set([m for m in matches if m.startswith('S2')])
            gt_s3_matches[s1] = set([m for m in matches if m.startswith('S3')])

random.seed(42)
sampled_s1_ids = random.sample(sampled_s1_ids, 25000)
sampled_s1_set = set(sampled_s1_ids)

required_s2_ids = set()
required_s3_ids = set()
for s1 in sampled_s1_ids:
    required_s2_ids.update(gt_s2_matches[s1])
    required_s3_ids.update(gt_s3_matches[s1])

print("Streaming Source 1...")
s1_rows = []
with open(os.path.join(data_dir, "train", "train_source1.tsv"), 'r', encoding='utf-8') as f:
    header = next(f).rstrip('\n').split('\t')
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if cols[0] in sampled_s1_set: s1_rows.append(cols)

print("Streaming Source 2...")
s2_rows = []
with open(os.path.join(data_dir, "train", "train_source2.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if cols[0] in required_s2_ids or random.random() < 0.05: s2_rows.append(cols)

print("Streaming Source 3...")
s3_rows = []
with open(os.path.join(data_dir, "train", "train_source3.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if cols[0] in required_s3_ids or random.random() < 0.05: s3_rows.append(cols)

df_s1 = pd.DataFrame(s1_rows, columns=header)
df_s2 = pd.DataFrame(s2_rows, columns=header)
df_s3 = pd.DataFrame(s3_rows, columns=header)

print("Normalizing...")
for df in [df_s1, df_s2, df_s3]:
    df['norm_name'] = df['business_name'].apply(normalize_text)
    df['norm_name_prefix'] = df['norm_name'].apply(lambda x: extract_prefix(x, 4))
    df['norm_name_token'] = df['norm_name'].apply(extract_token)
    df['norm_addr'] = df['business_address'].apply(normalize_text)
    df['norm_addr_token'] = df['norm_addr'].apply(extract_token)
    df['country_lower'] = df['country'].str.lower().str.strip()

def get_candidates(df1, df2, on_cols):
    start_time = time.time()
    v1 = df1
    v2 = df2
    for col in on_cols:
        v1 = v1[v1[col] != ""]
        v2 = v2[v2[col] != ""]
        
    cands = pd.merge(v1[['entity_id'] + on_cols], v2[['entity_id'] + on_cols], on=on_cols, suffixes=('_s1', '_s2'))
    cands_dict = cands.groupby('entity_id_s1')['entity_id_s2'].apply(set).to_dict()
    runtime = time.time() - start_time
    return cands_dict, runtime

def evaluate_candidates(cands_s2, cands_s3, runtime_total):
    total_cands_list = []
    
    found_s2_total = 0
    possible_s2_total = 0
    found_s3_total = 0
    possible_s3_total = 0
    
    s2_only_found, s2_only_pos = 0, 0
    s3_only_found, s3_only_pos = 0, 0
    both_found, both_pos = 0, 0
    
    for s1 in sampled_s1_ids:
        c2 = cands_s2.get(s1, set())
        c3 = cands_s3.get(s1, set())
        
        total_cands_list.append(len(c2) + len(c3))
        
        g2 = gt_s2_matches[s1]
        g3 = gt_s3_matches[s1]
        
        f2 = len(g2.intersection(c2))
        f3 = len(g3.intersection(c3))
        
        found_s2_total += f2
        possible_s2_total += len(g2)
        found_s3_total += f3
        possible_s3_total += len(g3)
        
        if len(g2) > 0 and len(g3) == 0:
            s2_only_found += f2
            s2_only_pos += len(g2)
        elif len(g3) > 0 and len(g2) == 0:
            s3_only_found += f3
            s3_only_pos += len(g3)
        elif len(g2) > 0 and len(g3) > 0:
            both_found += (f2 + f3)
            both_pos += (len(g2) + len(g3))
            
    avg_cands = np.mean(total_cands_list)
    med_cands = np.median(total_cands_list)
    p95_cands = np.percentile(total_cands_list, 95)
    max_cands = np.max(total_cands_list)
    
    return {
        "runtime_sec": round(runtime_total, 2),
        "total_cands": sum(total_cands_list),
        "avg_cands": round(avg_cands, 2),
        "med_cands": int(med_cands),
        "p95_cands": int(p95_cands),
        "max_cands": int(max_cands),
        "s2_recall": round(found_s2_total / possible_s2_total, 4) if possible_s2_total else 0,
        "s3_recall": round(found_s3_total / possible_s3_total, 4) if possible_s3_total else 0,
        "overall_recall": round((found_s2_total + found_s3_total) / (possible_s2_total + possible_s3_total), 4) if (possible_s2_total + possible_s3_total) else 0,
        "recall_s2_only": round(s2_only_found / s2_only_pos, 4) if s2_only_pos else 0,
        "recall_s3_only": round(s3_only_found / s3_only_pos, 4) if s3_only_pos else 0,
        "recall_both": round(both_found / both_pos, 4) if both_pos else 0
    }

rules = {
    "A": ["country_lower", "norm_name_prefix"],
    "B": ["country_lower", "norm_name_token"],
    "C": ["country_lower", "norm_addr_token"],
    "D": ["country_lower", "norm_name_token", "norm_addr_token"],
    "E": ["country_lower", "norm_name_prefix", "norm_addr_token"]
}

cands_s2 = {}
cands_s3 = {}
runtimes = {}

for rule_name, cols in rules.items():
    print(f"Generating candidates for Strategy {rule_name}...")
    c2, t2 = get_candidates(df_s1, df_s2, cols)
    c3, t3 = get_candidates(df_s1, df_s3, cols)
    cands_s2[rule_name] = c2
    cands_s3[rule_name] = c3
    runtimes[rule_name] = t2 + t3

results = {}

for r in ["A", "B", "C", "D", "E"]:
    print(f"Evaluating Strategy {r}...")
    results[f"Strategy {r}"] = evaluate_candidates(cands_s2[r], cands_s3[r], runtimes[r])

def union_dicts(dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k not in res: res[k] = set()
            res[k].update(v)
    return res

print("Evaluating Union A U B...")
u_s2 = union_dicts([cands_s2["A"], cands_s2["B"]])
u_s3 = union_dicts([cands_s3["A"], cands_s3["B"]])
t_total = runtimes["A"] + runtimes["B"]
results["Union A U B"] = evaluate_candidates(u_s2, u_s3, t_total)

print("Evaluating Union A U B U C...")
u_s2 = union_dicts([cands_s2["A"], cands_s2["B"], cands_s2["C"]])
u_s3 = union_dicts([cands_s3["A"], cands_s3["B"], cands_s3["C"]])
t_total += runtimes["C"]
results["Union A U B U C"] = evaluate_candidates(u_s2, u_s3, t_total)

print("Evaluating Union A U B U C U D...")
u_s2 = union_dicts([cands_s2["A"], cands_s2["B"], cands_s2["C"], cands_s2["D"]])
u_s3 = union_dicts([cands_s3["A"], cands_s3["B"], cands_s3["C"], cands_s3["D"]])
t_total += runtimes["D"]
results["Union A U B U C U D"] = evaluate_candidates(u_s2, u_s3, t_total)

print("Evaluating Union A U B U C U D U E...")
u_s2 = union_dicts([cands_s2["A"], cands_s2["B"], cands_s2["C"], cands_s2["D"], cands_s2["E"]])
u_s3 = union_dicts([cands_s3["A"], cands_s3["B"], cands_s3["C"], cands_s3["D"], cands_s3["E"]])
t_total += runtimes["E"]
results["Union A U B U C U D U E"] = evaluate_candidates(u_s2, u_s3, t_total)

with open("../outputs/blocking_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done!")
