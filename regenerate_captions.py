import os
import sys
import torch
import shutil
from src.dataset import TextDataset
from src.misc.config import cfg

def main():
    # Set up data directory
    data_dir = 'data/birds'
    
    # Create train and test directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)
    
    # Copy filenames.pickle to the correct location
    shutil.copy('data/train/filenames.pickle', os.path.join(data_dir, 'train/filenames.pickle'))
    shutil.copy('data/train/filenames.pickle', os.path.join(data_dir, 'test/filenames.pickle'))
    
    # Remove existing captions.pickle if it exists
    captions_path = os.path.join(data_dir, 'captions.pickle')
    if os.path.exists(captions_path):
        os.remove(captions_path)
    
    # Initialize dataset for both train and test splits
    print("Loading train dataset...")
    train_dataset = TextDataset(data_dir, split='train')
    print("Loading test dataset...")
    test_dataset = TextDataset(data_dir, split='test')
    
    print("Captions pickle file has been regenerated successfully!")

if __name__ == '__main__':
    main() 