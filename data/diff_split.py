import pandas as pd

# Read the data from the text file
df = pd.read_csv('clothing/clothing.inter', sep='\t')

# Function to assign x_label values for each user group
def assign_labels(group):
    # Sort interactions by timestamp
    group = group.sort_values('timestamp').reset_index(drop=True)
    n = len(group)
    if n >= 2:
        # Assign 0 to all but the last two interactions
        group.loc[:n-3, 'x_label'] = 0
        # Assign 1 to the second-to-last interaction
        group.loc[n-2, 'x_label'] = 1
        # Assign 2 to the last interaction
        group.loc[n-1, 'x_label'] = 2
    elif n == 1:
        # If only one interaction, assign it as test data
        group.loc[0, 'x_label'] = 2
    return group

# Apply the function to each user group
df_processed = df.groupby('userID', group_keys=False).apply(assign_labels)

# Save the processed DataFrame to a new file
df_processed.to_csv('clothing/clothing_diff_split.inter', sep='\t', index=False)