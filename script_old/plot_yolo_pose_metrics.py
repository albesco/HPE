
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_yolo_pose_metrics(csv_path, output_dir):
    """
    Plots the training and validation metrics from a YOLO pose training log.

    Args:
        csv_path (str): The path to the results.csv file.
        output_dir (str): The directory to save the plots in.
    """
    df = pd.read_csv(csv_path)
    
    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()

    # Calculate total validation loss
    val_loss_cols = ['val/box_loss', 'val/pose_loss', 'val/kobj_loss', 'val/cls_loss', 'val/dfl_loss', 'val/rle_loss']
    df['val/total_loss'] = df[val_loss_cols].sum(axis=1)

    # Create plots
    sns.set_theme(style="whitegrid")
    
    # Plot Loss
    plt.figure(figsize=(12, 6))
    df_filtered = df[df['epoch'] >= 3] # Filter epochs from 3 onwards
    plt.plot(df_filtered['epoch'], df_filtered['train/box_loss'], label='train/box_loss')
    plt.plot(df_filtered['epoch'], df_filtered['train/pose_loss'], label='train/pose_loss')
    plt.plot(df_filtered['epoch'], df_filtered['val/total_loss'], label='val/total_loss', linestyle='--')
    plt.plot(df_filtered['epoch'], df_filtered['val/box_loss'], label='val/box_loss', linestyle='--')
    plt.plot(df_filtered['epoch'], df_filtered['val/pose_loss'], label='val/pose_loss', linestyle='--')
    plt.title('YOLO Pose Training Losses (Epochs >= 3)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot mAP
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['metrics/mAP50-95(P)'], label='metrics/mAP50-95(P)')
    plt.plot(df['epoch'], df['metrics/mAP50(P)'], label='metrics/mAP50(P)', linestyle='--')
    plt.title('YOLO Pose mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    map_plot_path = os.path.join(output_dir, 'map_plot.png')
    plt.savefig(map_plot_path)
    plt.close()
    
    print(f"Plots saved to {loss_plot_path} and {map_plot_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot YOLO pose metrics from a results.csv file.')
    parser.add_argument('csv_path', type=str, help='The path to the results.csv file.')
    parser.add_argument('output_dir', type=str, help='The directory to save the plots in.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot_yolo_pose_metrics(args.csv_path, args.output_dir)
