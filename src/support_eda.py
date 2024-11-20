import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def distribution(df, first, last, col, n=1, size = (10, 5)):
    # Validate inputs
    if n <= 0:
        raise ValueError("Bin width (n) must be a positive integer.")
    if first >= last:
        raise ValueError("'first' must be less than 'last'.")
    
    # Define the bin edges to align with ticks
    bin_edges = np.arange(first, last + n, n)
    
    # Filter the data
    filtered_data = df[df[col].between(first, last)][col]
    
    if filtered_data.empty:
        print(f"No data available for the range {first} to {last}.")
        return

    # Set dynamic figure size based on the range
    plt.figure(figsize=size)
    
    # Create the histogram with aligned bins
    sns.histplot(filtered_data, bins=bin_edges, kde=False, color="skyblue", edgecolor="black")
    
    # Add title and labels
    plt.title(f"Distribution of {col} ({first}–{last})")
    plt.xlabel("")
    plt.ylabel("Frequency")
    
    # Set x-ticks to align with bins
    plt.xticks(bin_edges, rotation=45)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()