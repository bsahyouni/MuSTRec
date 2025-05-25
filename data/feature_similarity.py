import numpy as np


def compute_average_intra_modal_similarity_batched(file_path, batch_size=1000):
    """
    Computes average pairwise cosine similarity in a memory-efficient way using batches.

    Args:
    - file_path (str): Path to the .npy file containing features.
    - batch_size (int): Number of items per batch.

    Returns:
    - avg_sim (float): Average pairwise cosine similarity.
    """
    features = np.load(file_path)  # shape: (N, d)
    N = features.shape[0]

    if N < 2:
        return 1.0  # or 0.0, depending on the preference for single-item datasets

    # Normalize features (L2-normalization)
    norm_feats = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Initialize variables for accumulating similarity and pair counts
    total_sim = 0.0
    total_pairs = 0

    # Compute pairwise similarity in batches
    for i in range(0, N, batch_size):
        batch_feats = norm_feats[i:i + batch_size]  # Get a batch of features
        sim_matrix = batch_feats @ norm_feats.T  # Compute cosine similarity with all items

        # Exclude diagonal self-similarity for items within the same batch
        mask = np.ones_like(sim_matrix, dtype=bool)
        np.fill_diagonal(mask, False)

        total_sim += np.sum(sim_matrix[mask])  # Sum all pairwise similarities excluding diagonal
        total_pairs += mask.sum()  # Count the number of valid pairs

    avg_sim = total_sim / total_pairs
    return avg_sim

if __name__ == "__main__":
    # Baby dataset
    baby_text_feat = "baby/text_feat.npy"
    baby_image_feat = "baby/image_feat.npy"

    baby_text_sim = compute_average_intra_modal_similarity_batched(baby_text_feat)
    baby_image_sim = compute_average_intra_modal_similarity_batched(baby_image_feat)

    print(f"Baby - Average text–text similarity:  {baby_text_sim:.4f}")
    print(f"Baby - Average image–image similarity: {baby_image_sim:.4f}\n")

    # Elec dataset
    elec_text_feat = "elec-2/text_feat.npy"
    elec_image_feat = "elec/image_feat.npy"

    elec_text_sim = compute_average_intra_modal_similarity_batched(elec_text_feat)
    elec_image_sim = compute_average_intra_modal_similarity_batched(elec_image_feat)

    print(f"Electronics - Average text–text similarity:  {elec_text_sim:.4f}")
    print(f"Electronics - Average image–image similarity: {elec_image_sim:.4f}\n")

    # Sports dataset
    sports_text_feat = "sports/text_feat.npy"
    sports_image_feat = "sports/image_feat.npy"

    sports_text_sim = compute_average_intra_modal_similarity_batched(sports_text_feat)
    sports_image_sim = compute_average_intra_modal_similarity_batched(sports_image_feat)

    print(f"Sports - Average text–text similarity:  {sports_text_sim:.4f}")
    print(f"Sports - Average image–image similarity: {sports_image_sim:.4f}\n")