import torch
import torch.nn as nn
import torch.nn.functional as F

class MMRReranker:
    def __init__(self, lambda_mmr=0.5,top_k=20,n_items=500):
        self.lambda_mmr = lambda_mmr
        self.top_k = top_k
        self.n_items = n_items

    def rerank(self, user_e, all_item_e, scores):
        """
        Perform the MMR reranking based on relevance (score) and diversity (similarity).
        """
        topk_scores, topk_indices = torch.topk(scores, self.n_items, dim=1)
        print("topk_scores shape:", topk_scores.shape)
        print("topk_indices shape:", topk_indices.shape)

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
        remaining_items = topk_indices.tolist()

        # Pick first item (highest score)
        first_item = remaining_items.pop(0)
        selected_items.append(first_item)

        # Create dict for item and score of the item
        item_to_score = {item.item(): topk_scores[id].item() for id, item in enumerate(topk_indices)}

        for i in range(1, self.top_k):
            # Calculate relevance (score)
            relevance = torch.tensor([item_to_score[item] for item in remaining_items], device=all_item_e.device)

            # List to store similarity scores for each remaining item with selected items
            similarity_scores = []

            for id, item in enumerate(remaining_items):
                # Get the similarity between the current remaining item and all selected items
                selected_similarity = similarity_matrix[item, selected_items].sum()
                similarity_scores.append(selected_similarity)

            similarity_scores = torch.tensor(similarity_scores, device=all_item_e.device)

            # Compute MMR scores for each remaining item using relevance and similarity scores
            mmr_scores = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * similarity_scores

            # Select the next item with the highest MMR score
            next_item_index = torch.argmax(mmr_scores).item()
            next_item = remaining_items[next_item_index]

            # Add the selected item to the list and remove from remaining
            selected_items.append(next_item)
            remaining_items.remove(next_item)

        return selected_items

    def get_cosine_similarity(self, all_item_e):
        # Normalize the embeddings
        norm_item_e = F.normalize(all_item_e, p=2, dim=1)
        # Compute cosine similarity
        similarity_matrix = torch.matmul(norm_item_e, norm_item_e.T)
        return similarity_matrix

