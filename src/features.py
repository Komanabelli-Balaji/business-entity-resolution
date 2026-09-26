import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from src.normalize import tokenize

def token_jaccard(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def generate_features(df_pairs, df_s1, df_target):
    """
    df_pairs: DataFrame with 'entity_id_s1' and 'entity_id_s2'
    df_s1: S1 dataframe (must have entity_id indexed or mergeable)
    df_target: S2 or S3 dataframe
    """
    if df_pairs.empty:
        return pd.DataFrame()
        
    df = pd.merge(df_pairs, df_s1, left_on='entity_id_s1', right_on='entity_id', how='inner')
    df = pd.merge(df, df_target, left_on='entity_id_s2', right_on='entity_id', suffixes=('_1', '_2'), how='inner')
    
    features = pd.DataFrame()
    
    # 1. Name Features
    features['exact_name_match'] = (df['norm_name_1'] == df['norm_name_2']).astype(int)
    features['name_prefix_match'] = (df['norm_name_prefix_1'] == df['norm_name_prefix_2']).astype(int)
    
    # Fuzz ratio expects lists or strings; we'll use apply
    features['name_char_sim'] = df.apply(lambda row: fuzz.ratio(row['norm_name_1'], row['norm_name_2']), axis=1)
    
    # Pre-tokenize
    tok_name_1 = df['norm_name_1'].apply(tokenize)
    tok_name_2 = df['norm_name_2'].apply(tokenize)
    
    features['name_jaccard'] = [token_jaccard(t1, t2) for t1, t2 in zip(tok_name_1, tok_name_2)]
    features['name_len_diff'] = np.abs(df['norm_name_1'].str.len() - df['norm_name_2'].str.len())
    features['common_name_tokens'] = [len(set(t1) & set(t2)) for t1, t2 in zip(tok_name_1, tok_name_2)]
    
    # 2. Address Features
    features['exact_addr_match'] = (df['norm_addr_1'] == df['norm_addr_2']).astype(int)
    features['addr_missing'] = ((df['norm_addr_1'] == '') | (df['norm_addr_2'] == '')).astype(int)
    
    features['addr_char_sim'] = df.apply(lambda row: fuzz.ratio(row['norm_addr_1'], row['norm_addr_2']), axis=1)
    
    tok_addr_1 = df['norm_addr_1'].apply(tokenize)
    tok_addr_2 = df['norm_addr_2'].apply(tokenize)
    
    features['addr_jaccard'] = [token_jaccard(t1, t2) for t1, t2 in zip(tok_addr_1, tok_addr_2)]
    features['addr_len_diff'] = np.abs(df['norm_addr_1'].str.len() - df['norm_addr_2'].str.len())
    features['common_addr_tokens'] = [len(set(t1) & set(t2)) for t1, t2 in zip(tok_addr_1, tok_addr_2)]
    
    # 3. Country Match
    features['country_match'] = (df['country_lower_1'] == df['country_lower_2']).astype(int)
    
    # Retain IDs
    features['entity_id_s1'] = df['entity_id_s1']
    features['entity_id_s2'] = df['entity_id_s2']
    features['candidate_source'] = df['candidate_source']
    
    return features
