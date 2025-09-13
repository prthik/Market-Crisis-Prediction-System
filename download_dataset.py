import kagglehub
import os
import pandas as pd
import numpy as np

def download_stock_dataset():
    """Download the Kaggle stock market dataset"""
    print("Downloading dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("jakewright/9000-tickers-of-stock-market-data-full-history")
    
    print(f"Path to dataset files: {path}")
    
    # List files in the dataset directory
    print("\nDataset contents:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size:,} bytes)")
    
    return path

if __name__ == "__main__":
    dataset_path = download_stock_dataset()
