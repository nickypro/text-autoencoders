import pickle
import os
import gc
from tqdm import tqdm
import torch
from welford_torch import Welford
from utils_load_data import load_res_data, load_embeds
from dataclasses import dataclass

class Normalizer:
    def __init__(self, welford=None):
        self.welford = welford if welford is not None else Welford()

    def normalize(self, data):
        """Normalize data using precomputed mean and variance"""
        return (data - self.welford.mean) / torch.sqrt(self.welford.var_s + 1e-8)

    def restore(self, normed_data):
        """Restore normalized data to original scale using precomputed mean and variance"""
        return normed_data * torch.sqrt(self.welford.var_s + 1e-8) + self.welford.mean

    def add_all(self, data):
        """Add data to the underlying Welford statistics"""
        self.welford.add_all(data)

    def __call__(self, data):
        return self.normalize(data)

@dataclass
class WelfordData:
    welford_res: Welford
    welford_emb: Welford
    norm_res: Normalizer
    norm_emb: Normalizer

def load_or_compute_welford_stats(groups_to_load):
    """Load or compute Welford statistics for normalization"""
    os.makedirs('../data/welford_data', exist_ok=True)
    welford_file = f'../data/welford_data/welford_stats_10_{groups_to_load}.pkl'

    try:
        with open(welford_file, 'rb') as f:
            print("Loading existing welford data")
            welford_stats = pickle.load(f)
            welford_emb, welford_res = welford_stats['welford_emb'], welford_stats['welford_res']

    except FileNotFoundError:
        welford_emb = Welford()
        welford_res = Welford()

        for i in tqdm(range(10)):
            res_data = load_res_data(i, groups_to_load=groups_to_load).cuda()
            embeds = load_embeds(i).cuda()

            batch_size = 1000
            num_samples = res_data.size(0)
            for i in range(0, num_samples, batch_size):
                end_idx = i + batch_size
                batch_res = res_data[i:end_idx]
                batch_emb = embeds[i:end_idx]
                welford_res.add_all(batch_res)
                welford_emb.add_all(batch_emb)
            del res_data, embeds
            gc.collect()
            torch.cuda.empty_cache()

        # Save Welford statistics for first 10 files using pickle
        with open(welford_file, 'wb') as f:
            pickle.dump({
                'welford_emb': welford_emb,
                'welford_res': welford_res,
            }, f)

    return WelfordData(
        welford_res,
        welford_emb,
        Normalizer(welford_res),
        Normalizer(welford_emb)
    )
