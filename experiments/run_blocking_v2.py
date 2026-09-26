import pandas as pd
import numpy as np
import os
import json
import random
import re
import unicodedata
import time
from collections import defaultdict, Counter

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

def tokenize(text):
    if not text: return []
    return text.split()

print("Loading Data...")
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
    df['norm_addr'] = df['business_address'].apply(normalize_text)
    df['country_lower'] = df['country'].str.lower().str.strip()

print("Computing token frequencies...")
name_freqs = defaultdict(Counter)
addr_freqs = defaultdict(Counter)

for df in [df_s1, df_s2, df_s3]:
    for c, n, a in zip(df['country_lower'], df['norm_name'], df['norm_addr']):
        name_freqs[c].update(set(tokenize(n)))
        addr_freqs[c].update(set(tokenize(a)))

# Define a "rare" token as appearing <= MAX_FREQ times in our subset
# Given the subset is ~300k rows (approx 5-10% of total data),
# a freq of 100 corresponds to ~1000-2000 in the full dataset.
MAX_FREQ = 100

def get_rare_tokens(text, country, freqs_dict):
    tokens = set(tokenize(text))
    return [t for t in tokens if freqs_dict[country][t] <= MAX_FREQ]

print(f"Extracting rare tokens (MAX_FREQ={MAX_FREQ})...")
for df in [df_s1, df_s2, df_s3]:
    df['rare_name_tokens'] = [get_rare_tokens(n, c, name_freqs) for n, c in zip(df['norm_name'], df['country_lower'])]
    df['rare_addr_tokens'] = [get_rare_tokens(a, c, addr_freqs) for a, c in zip(df['norm_addr'], df['country_lower'])]

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

def get_exploded_candidates(df1, df2, list_col, match_cols):
    start_time = time.time()
    
    # Filter empty lists
    v1 = df1[df1[list_col].map(len) > 0]
    v2 = df2[df2[list_col].map(len) > 0]
    
    e1 = v1[['entity_id', 'country_lower', list_col]].explode(list_col)
    e2 = v2[['entity_id', 'country_lower', list_col]].explode(list_col)
    
    cands = pd.merge(e1, e2, on=match_cols, suffixes=('_s1', '_s2'))
    cands_dict = cands.groupby('entity_id_s1')['entity_id_s2'].apply(set).to_dict()
    
    runtime = time.time() - start_time
    return cands_dict, runtime

def get_standard_candidates(df1, df2, on_cols):
    start_time = time.time()
    v1 = df1
    v2 = df2
    for col in on_cols:
        v1 = v1[v1[col] != ""]
        v2 = v2[v2[col] != ""]
        
    cands = pd.merge(v1[['entity_id'] + on_cols], v2[['entity_id'] + on_cols], on=on_cols, suffixes=('_s1', '_s2'))
    cands_dict = cands.groupby('entity_id_s1')['entity_id_s2'].apply(set).to_dict()
    return cands_dict, time.time() - start_time

cands_s2 = {}
cands_s3 = {}
runtimes = {}

print("Evaluating 1. Country + Rare Name Token...")
c2, t2 = get_exploded_candidates(df_s1, df_s2, 'rare_name_tokens', ['country_lower', 'rare_name_tokens'])
c3, t3 = get_exploded_candidates(df_s1, df_s3, 'rare_name_tokens', ['country_lower', 'rare_name_tokens'])
cands_s2["RareName"] = c2
cands_s3["RareName"] = c3
runtimes["RareName"] = t2 + t3

print("Evaluating 2. Country + Rare Address Token...")
c2, t2 = get_exploded_candidates(df_s1, df_s2, 'rare_addr_tokens', ['country_lower', 'rare_addr_tokens'])
c3, t3 = get_exploded_candidates(df_s1, df_s3, 'rare_addr_tokens', ['country_lower', 'rare_addr_tokens'])
cands_s2["RareAddr"] = c2
cands_s3["RareAddr"] = c3
runtimes["RareAddr"] = t2 + t3

print("Evaluating 3. Country + Name Prefix (Baseline)...")
c2, t2 = get_standard_candidates(df_s1, df_s2, ['country_lower', 'norm_name_prefix'])
c3, t3 = get_standard_candidates(df_s1, df_s3, ['country_lower', 'norm_name_prefix'])
cands_s2["NamePrefix"] = c2
cands_s3["NamePrefix"] = c3
runtimes["NamePrefix"] = t2 + t3

results = {}

print("Scoring Strategy 1...")
results["Strategy 1 (Rare Name Token)"] = evaluate_candidates(cands_s2["RareName"], cands_s3["RareName"], runtimes["RareName"])

print("Scoring Strategy 2...")
results["Strategy 2 (Rare Addr Token)"] = evaluate_candidates(cands_s2["RareAddr"], cands_s3["RareAddr"], runtimes["RareAddr"])

def union_dicts(dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k not in res: res[k] = set()
            res[k].update(v)
    return res

print("Scoring Union 1 U 2...")
u_s2 = union_dicts([cands_s2["RareName"], cands_s2["RareAddr"]])
u_s3 = union_dicts([cands_s3["RareName"], cands_s3["RareAddr"]])
results["Union 1 U 2"] = evaluate_candidates(u_s2, u_s3, runtimes["RareName"] + runtimes["RareAddr"])

print("Scoring Union 1 U 2 U NamePrefix...")
u_s2 = union_dicts([cands_s2["RareName"], cands_s2["RareAddr"], cands_s2["NamePrefix"]])
u_s3 = union_dicts([cands_s3["RareName"], cands_s3["RareAddr"], cands_s3["NamePrefix"]])
results["Union 1 U 2 U NamePrefix"] = evaluate_candidates(u_s2, u_s3, runtimes["RareName"] + runtimes["RareAddr"] + runtimes["NamePrefix"])

with open("../outputs/blocking_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done!")
