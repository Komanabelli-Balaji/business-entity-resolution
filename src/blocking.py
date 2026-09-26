import pandas as pd
import time
from collections import defaultdict, Counter
from src.normalize import tokenize

class FrequencyBlocker:
    def __init__(self, max_freq=1500):
        self.max_freq = max_freq
        self.name_freqs = defaultdict(Counter)
        self.addr_freqs = defaultdict(Counter)
        self.df_s2 = None
        self.df_s3 = None
        
    def fit_frequencies(self, df):
        """Update token counts from a dataframe"""
        for c, n, a in zip(df['country_lower'], df['norm_name'], df['norm_addr']):
            if c:
                self.name_freqs[c].update(set(tokenize(n)))
                self.addr_freqs[c].update(set(tokenize(a)))
                
    def get_rare_tokens(self, text, country, freqs_dict):
        if not country or country not in freqs_dict:
            return []
        tokens = set(tokenize(text))
        return [t for t in tokens if freqs_dict[country][t] <= self.max_freq]

    def prep_targets(self, df_s2, df_s3):
        """Pre-compute rare tokens on the target datasets"""
        print("Preparing target sources for blocking (computing rare tokens)...")
        for df in [df_s2, df_s3]:
            if 'rare_name_tokens' not in df.columns:
                df['rare_name_tokens'] = [self.get_rare_tokens(n, c, self.name_freqs) for n, c in zip(df['norm_name'], df['country_lower'])]
            if 'rare_addr_tokens' not in df.columns:
                df['rare_addr_tokens'] = [self.get_rare_tokens(a, c, self.addr_freqs) for a, c in zip(df['norm_addr'], df['country_lower'])]
        self.df_s2 = df_s2
        self.df_s3 = df_s3

    def _get_exploded_candidates(self, df1, df2, list_col, match_cols):
        v1 = df1[df1[list_col].map(len) > 0]
        v2 = df2[df2[list_col].map(len) > 0]
        if v1.empty or v2.empty: return pd.DataFrame(columns=['entity_id_s1', 'entity_id_s2'])
        
        e1 = v1[['entity_id', 'country_lower', list_col]].explode(list_col)
        e2 = v2[['entity_id', 'country_lower', list_col]].explode(list_col)
        
        cands = pd.merge(e1, e2, on=match_cols, suffixes=('_s1', '_s2'))
        return cands[['entity_id_s1', 'entity_id_s2']].drop_duplicates()

    def _get_standard_candidates(self, df1, df2, on_cols):
        v1 = df1
        v2 = df2
        for col in on_cols:
            v1 = v1[v1[col] != ""]
            v2 = v2[v2[col] != ""]
        if v1.empty or v2.empty: return pd.DataFrame(columns=['entity_id_s1', 'entity_id_s2'])
            
        cands = pd.merge(v1[['entity_id'] + on_cols], v2[['entity_id'] + on_cols], on=on_cols, suffixes=('_s1', '_s2'))
        return cands[['entity_id_s1', 'entity_id_s2']].drop_duplicates()

    def generate_candidates(self, df_s1_batch):
        """Generate candidates for a batch of S1 records."""
        # Add rare tokens to S1
        df_s1_batch['rare_name_tokens'] = [self.get_rare_tokens(n, c, self.name_freqs) for n, c in zip(df_s1_batch['norm_name'], df_s1_batch['country_lower'])]
        df_s1_batch['rare_addr_tokens'] = [self.get_rare_tokens(a, c, self.addr_freqs) for a, c in zip(df_s1_batch['norm_addr'], df_s1_batch['country_lower'])]
        
        all_pairs = []
        for df_target, source_label in [(self.df_s2, 'S2'), (self.df_s3, 'S3')]:
            c_name = self._get_exploded_candidates(df_s1_batch, df_target, 'rare_name_tokens', ['country_lower', 'rare_name_tokens'])
            c_addr = self._get_exploded_candidates(df_s1_batch, df_target, 'rare_addr_tokens', ['country_lower', 'rare_addr_tokens'])
            c_pref = self._get_standard_candidates(df_s1_batch, df_target, ['country_lower', 'norm_name_prefix'])
            
            cands = pd.concat([c_name, c_addr, c_pref]).drop_duplicates()
            if not cands.empty:
                cands['candidate_source'] = source_label
                all_pairs.append(cands)
                
        if all_pairs:
            return pd.concat(all_pairs, ignore_index=True)
        return pd.DataFrame(columns=['entity_id_s1', 'entity_id_s2', 'candidate_source'])
