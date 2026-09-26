import pandas as pd
import numpy as np
import random
from sklearn.ensemble import HistGradientBoostingClassifier

class PairMatcher:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.1)
        self.feature_cols = [
            'exact_name_match', 'name_prefix_match', 'name_char_sim', 
            'name_jaccard', 'name_len_diff', 'common_name_tokens',
            'exact_addr_match', 'addr_missing', 'addr_char_sim', 
            'addr_jaccard', 'addr_len_diff', 'common_addr_tokens',
            'country_match'
        ]
        
    def prepare_training_data(self, features_df, ground_truth_dict):
        """
        features_df: DataFrame with features, entity_id_s1, entity_id_s2
        ground_truth_dict: {s1_id: set(s2_and_s3_ids)}
        """
        labels = []
        for s1, s2 in zip(features_df['entity_id_s1'], features_df['entity_id_s2']):
            matches = ground_truth_dict.get(s1, set())
            labels.append(1 if s2 in matches else 0)
            
        features_df['label'] = labels
        return features_df
        
    def fit(self, df_train):
        X = df_train[self.feature_cols]
        y = df_train['label']
        print(f"Training on {len(X)} pairs ({y.sum()} positive, {len(y) - y.sum()} negative)")
        self.model.fit(X, y)
        
    def predict_proba(self, df):
        X = df[self.feature_cols]
        probs = self.model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        else:
            return np.zeros(len(X))
