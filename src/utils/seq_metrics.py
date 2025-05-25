import math

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def mean_average_precision_at_k(actual, predicted, topk):
    """
    Computes the MAP@K for a list of users.

    Parameters:
    - actual: List of actual relevant items (one per user).
    - predicted: List of lists of predicted items per user.
    - topk: The cutoff rank K.

    Returns:
    - MAP@K averaged over all users.
    """
    sum_ap = 0.0
    num_users = len(actual)
    for i in range(num_users):
        actual_item = actual[i]
        predicted_items = predicted[i][:topk]
        ap = 0.0
        for j, pred_item in enumerate(predicted_items):
            if pred_item == actual_item:
                # Precision at the rank where the relevant item is found
                ap = 1.0 / (j + 1)
                break  # Stop after finding the first relevant item
        sum_ap += ap
    return sum_ap / num_users


def precision_at_k(actual, predicted, topk):
    """
    Computes the Precision@K for a list of users.

    Parameters:
    - actual: List of actual relevant items (one per user).
    - predicted: List of lists of predicted items per user.
    - topk: The cutoff rank K.

    Returns:
    - Precision@K averaged over all users.
    """
    sum_precision = 0.0
    num_users = len(actual)
    for i in range(num_users):
        actual_item = actual[i]
        predicted_items = predicted[i][:topk]
        if actual_item in predicted_items:
            sum_precision += 1.0 / topk  # Since there's only one relevant item
        else:
            sum_precision += 0.0
    return sum_precision / num_users
