
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_csv_data(csv_file, x_column, y_column):
    """
    Reads a CSV file and creates a plot of y_column vs. x_column.

    Args:
        csv_file (str): The path to the CSV file.
        x_column (str): The name of the column to use for the x-axis.
        y_column (str): The name of the column to use for the y-axis.
    """
    output_dir = os.path.dirname(csv_file)
    base_name = os.path.basename(csv_file).replace('.csv', '')
    output_file = os.path.join(output_dir, f'{base_name}_{y_column.replace("/", "-")}_vs_{x_column}.png')

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()

    # Check if the columns exist
    if x_column not in df.columns:
        print(f"Error: Column '{x_column}' not found in {csv_file}")
        return
    if y_column not in df.columns:
        print(f"Error: Column '{y_column}' not found in {csv_file}")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o')
    plt.title(f'{y_column} vs. {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)

    # Save the plot
    plt.savefig(output_file)

    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument('csv_file', type=str, help='The path to the CSV file.')
    parser.add_argument('--x_column', type=str, default='epoch', help='The column for the x-axis.')
    parser.add_argument('--y_column', type=str, default='metrics/mAP50-95(P)', help='The column for the y-axis.')
    args = parser.parse_args()

    plot_csv_data(args.csv_file, args.x_column, args.y_column)
