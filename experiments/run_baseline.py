import os
import sys
import pandas as pd
import numpy as np
import json
import random
import time
import subprocess

from src.normalize import normalize_text, extract_prefix
from src.blocking import FrequencyBlocker
from src.features import generate_features
from src.model import PairMatcher

import boto3
import io

S3_BUCKET = "hackathon-machines"
S3_PREFIX = "entity-resolution/dataset/"
OUTPUT_DIR = "/home/balaji/Projects/entity-resolution/business_entity_resolution/outputs"

s3 = boto3.client("s3", region_name="ap-south-1")

def get_s3_obj(path):
    key = S3_PREFIX + path
    print(f"Loading s3://{S3_BUCKET}/{key}...")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return io.BytesIO(obj['Body'].read())

def load_and_normalize(path):
    f = get_s3_obj(path)
    df = pd.read_csv(f, sep='\t', dtype=str, engine='c').fillna("")
    if 'business_name' in df.columns:
        df['norm_name'] = df['business_name'].apply(normalize_text)
        df['norm_name_prefix'] = df['norm_name'].apply(lambda x: extract_prefix(x, 4))
    if 'business_address' in df.columns:
        df['norm_addr'] = df['business_address'].apply(normalize_text)
    if 'country' in df.columns:
        df['country_lower'] = df['country'].str.lower().str.strip()
    return df

def get_ground_truth(path):
    gt = {}
    f = get_s3_obj(path)
    # read line by line from bytes
    first = True
    for line in f:
        line = line.decode('utf-8').rstrip('\n')
        if first:
            first = False
            continue
        parts = line.split('\t')
        s1 = parts[0]
        if len(parts) > 1 and parts[1].strip():
            gt[s1] = set([x.strip() for x in parts[1].split(',') if x.strip()])
        else:
            gt[s1] = set()
    return gt


def compute_metrics(true_pos, false_pos, false_neg):
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f05

def main():
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Training Data
    print("\\n--- TRAINING PHASE ---")
    train_s2 = load_and_normalize(os.path.join(DATA_DIR, "train", "train_source2.tsv"))
    train_s3 = load_and_normalize(os.path.join(DATA_DIR, "train", "train_source3.tsv"))
    train_gt = get_ground_truth(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    
    # Load a sample of S1 for training (e.g., 50k rows)
    # We will split 40k for training, 10k for validation
    train_s1_full = load_and_normalize(os.path.join(DATA_DIR, "train", "train_source1.tsv"))
    s1_ids = list(train_s1_full['entity_id'])
    random.shuffle(s1_ids)
    
    sample_size = min(50000, len(s1_ids))
    train_sample_ids = set(s1_ids[:sample_size])
    df_s1_sample = train_s1_full[train_s1_full['entity_id'].isin(train_sample_ids)].copy()
    
    train_ids = set(s1_ids[:int(sample_size * 0.8)])
    val_ids = set(s1_ids[int(sample_size * 0.8):sample_size])
    
    df_train_s1 = df_s1_sample[df_s1_sample['entity_id'].isin(train_ids)].copy()
    df_val_s1 = df_s1_sample[df_s1_sample['entity_id'].isin(val_ids)].copy()
    
    print(f"Sampled {len(df_train_s1)} S1 for train, {len(df_val_s1)} S1 for val")
    
    # Initialize and fit Train Blocker
    train_blocker = FrequencyBlocker(max_freq=500)
    train_blocker.fit_frequencies(train_s2)
    train_blocker.fit_frequencies(train_s3)
    train_blocker.prep_targets(train_s2, train_s3)
    
    # Generate Training Candidates
    print("Generating training candidates...")
    train_cands = train_blocker.generate_candidates(df_train_s1)
    print(f"Generated {len(train_cands)} raw candidates for training.")
    
    # Generate Training Features
    print("Generating training features...")
    df_train_cands_s2 = train_cands[train_cands['candidate_source'] == 'S2']
    df_train_cands_s3 = train_cands[train_cands['candidate_source'] == 'S3']
    
    feat_train_s2 = generate_features(df_train_cands_s2, df_train_s1, train_s2)
    feat_train_s3 = generate_features(df_train_cands_s3, df_train_s1, train_s3)
    feat_train = pd.concat([feat_train_s2, feat_train_s3], ignore_index=True)
    
    # Label and downsample negatives
    matcher = PairMatcher()
    feat_train = matcher.prepare_training_data(feat_train, train_gt)
    
    positives = feat_train[feat_train['label'] == 1]
    negatives = feat_train[feat_train['label'] == 0]
    # Sample negatives (e.g. 5x positives)
    n_neg = min(len(negatives), len(positives) * 5)
    negatives_sampled = negatives.sample(n=n_neg, random_state=42)
    feat_train_balanced = pd.concat([positives, negatives_sampled], ignore_index=True)
    
    # Train model
    matcher.fit(feat_train_balanced)
    
    # Free up memory
    del train_cands, df_train_cands_s2, df_train_cands_s3, feat_train_s2, feat_train_s3, feat_train
    
    # Validation Phase
    print("\\n--- VALIDATION PHASE ---")
    val_cands = train_blocker.generate_candidates(df_val_s1)
    
    df_val_cands_s2 = val_cands[val_cands['candidate_source'] == 'S2']
    df_val_cands_s3 = val_cands[val_cands['candidate_source'] == 'S3']
    
    feat_val_s2 = generate_features(df_val_cands_s2, df_val_s1, train_s2)
    feat_val_s3 = generate_features(df_val_cands_s3, df_val_s1, train_s3)
    feat_val = pd.concat([feat_val_s2, feat_val_s3], ignore_index=True)
    feat_val = matcher.prepare_training_data(feat_val, train_gt)
    
    feat_val['prob'] = matcher.predict_proba(feat_val)
    
    # Threshold selection
    best_f05 = -1
    best_threshold = 0.5
    val_metrics = {}
    
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        preds = feat_val[feat_val['prob'] >= th]
        
        # Calculate metrics for the validation set (by evaluating exactly what we'd output)
        pred_dict = preds.groupby('entity_id_s1')['entity_id_s2'].apply(set).to_dict()
        
        true_pos = 0
        false_pos = 0
        false_neg = 0
        
        for s1 in val_ids:
            gt_matches = train_gt.get(s1, set())
            pd_matches = pred_dict.get(s1, set())
            
            true_pos += len(gt_matches & pd_matches)
            false_pos += len(pd_matches - gt_matches)
            false_neg += len(gt_matches - pd_matches)
            
        p, r, f = compute_metrics(true_pos, false_pos, false_neg)
        val_metrics[th] = {"Precision": p, "Recall": r, "F0.5": f}
        
        if f > best_f05:
            best_f05 = f
            best_threshold = th
            
    print(f"Selected Threshold: {best_threshold} (F0.5: {best_f05:.4f})")
    
    # Free up memory
    del train_s2, train_s3, train_blocker, df_train_s1, df_val_s1, val_cands, feat_val
    
    # 2. Test Phase
    print("\\n--- TEST PHASE ---")
    test_s2 = load_and_normalize(os.path.join(DATA_DIR, "test", "test_source2.tsv"))
    test_s3 = load_and_normalize(os.path.join(DATA_DIR, "test", "test_source3.tsv"))
    test_s1_full = load_and_normalize(os.path.join(DATA_DIR, "test", "test_source1.tsv"))
    
    test_blocker = FrequencyBlocker(max_freq=500)
    test_blocker.fit_frequencies(test_s2)
    test_blocker.fit_frequencies(test_s3)
    test_blocker.prep_targets(test_s2, test_s3)
    
    predictions = {}
    
    chunk_size = 50000
    n_chunks = int(np.ceil(len(test_s1_full) / chunk_size))
    
    print(f"Processing {len(test_s1_full)} test records in {n_chunks} chunks...")
    
    total_test_candidates = 0
    
    for i in range(n_chunks):
        print(f"Processing Chunk {i+1}/{n_chunks}...")
        df_chunk = test_s1_full.iloc[i*chunk_size : (i+1)*chunk_size].copy()
        
        cands = test_blocker.generate_candidates(df_chunk)
        total_test_candidates += len(cands)
        
        if cands.empty:
            for s1 in df_chunk['entity_id']:
                predictions[s1] = []
            continue
            
        cands_s2 = cands[cands['candidate_source'] == 'S2']
        cands_s3 = cands[cands['candidate_source'] == 'S3']
        
        feat_s2 = generate_features(cands_s2, df_chunk, test_s2)
        feat_s3 = generate_features(cands_s3, df_chunk, test_s3)
        feat = pd.concat([feat_s2, feat_s3], ignore_index=True)
        
        if not feat.empty:
            feat['prob'] = matcher.predict_proba(feat)
            preds = feat[feat['prob'] >= best_threshold]
            
            # Group by s1 and collect sorted matches
            for s1, group in preds.groupby('entity_id_s1'):
                matches = sorted(list(set(group['entity_id_s2'])))
                predictions[s1] = matches
                
        # Fill in singletons
        for s1 in df_chunk['entity_id']:
            if s1 not in predictions:
                predictions[s1] = []
                
    # 3. Output Generation
    print("\\n--- GENERATING SUBMISSION ---")
    out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\\tmatched_entity_ids\\n")
        # Ensure we write exactly all test S1 IDs in order
        for s1 in test_s1_full['entity_id']:
            matches = predictions.get(s1, [])
            f.write(f"{s1}\\t{','.join(matches)}\\n")
            
    print("Saved matching_results.tsv.")
    
    # Save Metrics
    metrics_log = {
        "training_positive_pairs": int(positives.shape[0]),
        "training_negative_pairs": int(negatives_sampled.shape[0]),
        "validation_pairs": int(feat_val.shape[0]),
        "test_candidate_pairs": int(total_test_candidates),
        "selected_threshold": float(best_threshold),
        "validation_metrics": val_metrics[best_threshold],
        "predicted_matches": sum(len(v) for v in predictions.values()),
        "predicted_singletons": sum(1 for v in predictions.values() if len(v) == 0)
    }
    
    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
        
    print("\\nBASELINE COMPLETE")
    print(f"Training pairs:\\n  Positive: {metrics_log['training_positive_pairs']}\\n  Negative: {metrics_log['training_negative_pairs']}")
    print(f"Validation:\\n  Precision: {metrics_log['validation_metrics']['Precision']:.4f}\\n  Recall: {metrics_log['validation_metrics']['Recall']:.4f}\\n  F0.5: {metrics_log['validation_metrics']['F0.5']:.4f}\\n  Selected threshold: {best_threshold}")
    print(f"Test:\\n  Source-1 entities: {len(test_s1_full)}\\n  Candidate pairs: {metrics_log['test_candidate_pairs']}\\n  Predicted matches: {metrics_log['predicted_matches']}\\n  Predicted singletons: {metrics_log['predicted_singletons']}")
    
    print("\\nRunning Validator...")
    val_cmd = [sys.executable, "/home/balaji/Projects/entity-resolution/student_resource/utils/validate_submission.py", "--submission", out_path]
    try:
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("Validator: PASS")
        else:
            print("Validator: FAIL")
            print(res.stderr)
    except Exception as e:
        print(f"Validator: FAIL ({str(e)})")

if __name__ == "__main__":
    main()
