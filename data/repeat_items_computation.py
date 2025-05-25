import pandas as pd

def compute_normalized_repeats_and_find_repeat_users(file_path):
    # Load the file
    df = pd.read_csv(file_path, sep='\t', header=0)

    # Group by user
    grouped = df.groupby('userID')

    user_normalized_repeats = []
    users_with_repeats = []  # List to store users who have repeat items

    for user, group in grouped:
        # Separate items by x_label
        train_items = set(group.loc[group['x_label'] == 0, 'itemID'])
        val_items   = set(group.loc[group['x_label'] == 1, 'itemID'])
        test_items  = set(group.loc[group['x_label'] == 2, 'itemID'])

        # Combine val+test items
        vt_items = val_items.union(test_items)

        if len(vt_items) == 0:
            # If no validation or test items, treat the normalized repeat as 0 or skip
            user_normalized_repeats.append(0.0)
            continue

        # Count how many val+test items appear in the training set
        repeat_count = sum((item in train_items) for item in vt_items)

        # If there are any repeat items, record the user
        if repeat_count > 0:
            users_with_repeats.append(user)

        # Compute normalized repeat for this user
        norm_repeat = repeat_count / float(len(vt_items))
        user_normalized_repeats.append(norm_repeat)

    # Compute the average normalized repeat across all users
    if len(user_normalized_repeats) == 0:
        average_norm_repeat = 0.0
    else:
        average_norm_repeat = sum(user_normalized_repeats) / len(user_normalized_repeats)

    return average_norm_repeat, users_with_repeats

# Example usage:
baby_avg, baby_repeat_users = compute_normalized_repeats_and_find_repeat_users("baby/baby_diff_split.inter")
print("Average normalized repeat - Baby   :", baby_avg)
print("Users with repeats - Baby          :", baby_repeat_users)

elec_avg, elec_repeat_users = compute_normalized_repeats_and_find_repeat_users("elec/elec_diff_split.inter")
print("Average normalized repeat - Elec   :", elec_avg)
print("Users with repeats - Elec          :", elec_repeat_users)

sport_avg, sport_repeat_users = compute_normalized_repeats_and_find_repeat_users("sports/sports_diff_split.inter")
print("Average normalized repeat - Sports :", sport_avg)
print("Users with repeats - Sports        :", sport_repeat_users)