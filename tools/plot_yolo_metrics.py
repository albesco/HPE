import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(csv_path, x_column, y_column, output_path):
    """
    Plots a metric from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        x_column (str): Name of the column for the x-axis.
        y_column (str): Name of the column for the y-axis.
        output_path (str): Path to save the output PNG file.
    """
    df = pd.read_csv(csv_path)
    
    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o')
    plt.title(f'{y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot metrics from a CSV file.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--x_column', type=str, required=True, help='Column for the x-axis.')
    parser.add_argument('--y_column', type=str, required=True, help='Column for the y-axis.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output PNG file.')
    
    args = parser.parse_args()
    
    plot_metrics(args.csv_path, args.x_column, args.y_column, args.output_path)
