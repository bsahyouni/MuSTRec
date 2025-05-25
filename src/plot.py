import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = "Summary_of_Results_Used_for_Graphs.csv"  # Ensure the file is in the same directory or provide the full path
summary_df = pd.read_csv(file_path)
summary_df["Model"] = summary_df["Model"].replace("BSARec", "MuSTRec-B")

# Define colors and patterns
colors = ['#ADD8E6', '#90EE90', '#F08080', '#FFA07A', '#DDA0DD', '#F5DEB3', '#D3D3D3']
patterns = ['/', '\\', '|', '-', '+', 'x', '*']

# Define axis limits
hr_limits = {
    "Clothing": 0.09,
    "Sports": 0.11,
    "Electricity": 0.09,
    "Baby": 0.10
}

ndcg_limits = {
    "Clothing": 0.045,
    "Sports": 0.055,
    "Electricity": 0.045,
    "Baby": 0.05
}

# Iterate through datasets and generate graphs
for dataset_name, dataset_df in summary_df.groupby("Dataset"):
    # Extract necessary data
    models = dataset_df["Model"].tolist()
    hr_values = dataset_df["HR@20"].tolist()
    ndcg_values = dataset_df["N@20"].tolist()

    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Define bar positions
    bar_width = 1.0
    x_positions_hr = np.arange(len(models))
    x_positions_ndcg = x_positions_hr + len(models) + 1

    # Plot HR@20
    for i, (model, color, pattern) in enumerate(zip(models, colors, patterns)):
        ax1.bar(
            x_positions_hr[i], hr_values[i],
            width=bar_width,
            color=color, edgecolor='black',
            hatch=pattern, label=model
        )

    # Configure primary y-axis
    ax1.set_ylabel('HR@20', fontsize=30)
    ax1.set_ylim(0, hr_limits[dataset_name])
    ax1.tick_params(axis='both', which='major', labelsize=30)

    # Create secondary y-axis for NDCG@20
    ax2 = ax1.twinx()

    # Plot NDCG@20
    for i, (model, color, pattern) in enumerate(zip(models, colors, patterns)):
        ax2.bar(
            x_positions_ndcg[i], ndcg_values[i],
            width=bar_width,
            color=color, edgecolor='black',
            hatch=pattern
        )

    # Configure secondary y-axis
    ax2.set_ylabel('NDCG@20', fontsize=30)
    ax2.set_ylim(0, ndcg_limits[dataset_name])
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.yaxis.set_label_position('right')

    # Set labels
    x_combined_positions = [len(models) / 2 - 0.5, len(models) + len(models) / 2 + 0.5]
    ax1.set_xticks(x_combined_positions)
    ax1.set_xticklabels(['HR@20', 'NDCG@20'], fontsize=30)

    # Add gridlines
    ax1.grid(axis='y', linestyle='--', linewidth=0.7)
    ax1.grid(axis='x', linestyle='', linewidth=0)

    # Add solid lines for the top and right chart area boundaries
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_linewidth(1.5)
        ax1.spines[spine].set_color('black')

    # Add a single shared legend with all model names from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine legends from both axes
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Only add legend if there are handles (ensures it gets displayed)
    if handles:
        plt.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),  # Move legend further up
            fontsize=22,
            ncol=4,
            #title="Models",
            #title_fontsize=26
        )

    #plt.subplots_adjust(top=0.65)
    plt.tight_layout()
    #plt.title(f"{dataset_name} Dataset", fontsize=32)
    plt.show()
