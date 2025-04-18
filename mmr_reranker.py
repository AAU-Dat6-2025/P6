import math
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MMRReranker:
    def __init__(self,lambda_mmr=0.5, top_k=20,n_items=500):
        self.lambda_mmr = lambda_mmr
        self.top_k = top_k
        self.n_items = n_items

    def rerank(self, user_e, all_item_e, scores):
        """
        Perform the MMR reranking based on relevance (score) and diversity (similarity).
        """
        topk_scores, topk_indices = torch.topk(scores, self.n_items, dim=1)

        similarity_matrix = self.get_cosine_similarity(all_item_e)

        all_mmr_indices = []

        # Process each user
        for u in range(user_e.shape[0]):
            print(f"\n Processing user {u}")
            mmr_topk = self.compute_mmr(
                similarity_matrix,
                all_item_e,
                topk_scores[u],
                topk_indices[u]
            )
            all_mmr_indices.append(mmr_topk)

        return all_mmr_indices

    def compute_mmr(self, similarity_matrix, all_item_e, topk_scores, topk_indices):
        selected_items = []
        full_list = topk_indices.tolist()

        # Pick first item (highest score)
        first_item = full_list.pop(0)
        selected_items.append(first_item)

        # Create dict for item and score of the item
        item_to_score = {item.item(): topk_scores[id].item() for id, item in enumerate(topk_indices)}

        while len(selected_items) < self.top_k:
            # Calculate relevance (score)
            relevance = torch.tensor([item_to_score[item] for item in full_list], device=all_item_e.device)

            # List to store similarity scores for each remaining item with selected items
            similarity_scores = []

            for id, item in enumerate(full_list):
                # Get the similarity between the current remaining item and all selected items
                selected_similarity = similarity_matrix[item, selected_items].sum()
                similarity_scores.append(selected_similarity)

            similarity_scores = torch.tensor(similarity_scores, device=all_item_e.device)

            # Compute MMR scores for each remaining item using relevance and similarity scores
            mmr_scores = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * similarity_scores

            # Select the next item with the highest MMR score
            next_item_index = torch.argmax(mmr_scores).item()
            next_item = full_list[next_item_index]

            # Add the selected item to the list and remove from remaining
            selected_items.append(next_item)
            full_list.remove(next_item)

        return selected_items

    def get_cosine_similarity(self, all_item_e):
        norm_item_e = F.normalize(all_item_e, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_item_e, norm_item_e.transpose(0, 1))
        return similarity_matrix
