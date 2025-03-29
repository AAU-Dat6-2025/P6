r"""
BPR with Zipf's Penalty
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
    Zipf’s Law Penalty added to discourage over-recommendation of popular items. "Zipf Matrix Factorization : Matrix Factorization with
    Matthew Effect Reduction" in RecSys 2021.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPRZipf(GeneralRecommender):
    r"""
    BPR with Zipf's penalty to reduce popularity bias.
    Math for Zipf's Penalty:
    s=1+n(\sum_{i=1}^{n}ln(\frac{x_i}{x_{max}}))^{-1}
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPRZipf, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.zipf_alpha = config["zipf_alpha"]  # Strength of Zipf's penalty this needs to be tweaked somehow to find the optimal. ie make the factor global and tweak it during training

        # Define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # Compute item popularity for Zipf's penalty
        self.build_item_popularity(dataset)

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def build_item_popularity(self, dataset):
        """Precompute item popularity based on dataset interactions."""
        item_counts = torch.zeros(self.n_items, device=self.device)

        # Iterate through all interactions in the dataset
        item_ids = dataset.inter_feat[self.ITEM_ID]

        # Count occurrences of each item
        for item_id in item_ids:
            item_counts[item_id] += 1

        # Normalize popularity to avoid extreme values
        self.item_popularity = item_counts / item_counts.sum()  # Calculate x_i / x_max
        self.item_popularity = self.item_popularity.clamp(min=1e-6)  # Avoid log(0)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        
        # Compute standard BPR loss
        loss = self.loss(pos_item_score, neg_item_score)
        
        # Apply Zipf's penalty: log-scaled popularity discourages recommending frequent items
        zipf_penalty = self.zipf_alpha * (
            torch.log1p(self.item_popularity[pos_item]) + torch.log1p(self.item_popularity[neg_item])
        )
        loss += torch.mean(zipf_penalty)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))

        # Apply Zipf’s penalty to final predictions
        zipf_penalty = self.zipf_alpha * torch.log1p(self.item_popularity)
        score -= zipf_penalty  # Subtract penalty from score

        return score.view(-1)
