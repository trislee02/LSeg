import os 
import shutil
import argparse
import numpy as np
import re

"""
Split seen Grasp-Anything++ dataset into train and validation sets.
Usage: python prepare_grasp_anything.py --seen-dir ../grasp-anything/seen --split-ratio 0.8 --shuffle True
""" 

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split seen Grasp-Anything++ dataset.',
        epilog='Example: python prepare_ade20k.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seen-dir', default=None, help='seen dataset directory on disk')
    parser.add_argument('--split-ratio', default=0.8, help='train/val split ratio')
    parser.add_argument('--shuffle', default=True, help='shuffle indices')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    train_path = os.path.join(args.seen_dir, 'train')
    val_path = os.path.join(args.seen_dir, 'val')

    grasp_instructions_dir = os.path.join(args.seen_dir, 'grasp_instructions')
    grasp_label_dir = os.path.join(args.seen_dir, 'grasp_label')
    images_dir = os.path.join(args.seen_dir, 'image')

    train_grasp_instructions_dir = os.path.join(train_path, 'grasp_instructions')
    train_grasp_label_dir = os.path.join(train_path, 'grasp_label')
    train_images_dir = os.path.join(train_path, 'image')

    val_grasp_instructions_dir = os.path.join(val_path, 'grasp_instructions')
    val_grasp_label_dir = os.path.join(val_path, 'grasp_label')
    val_images_dir = os.path.join(val_path, 'image')

    grasp_instructions_files = os.listdir(grasp_instructions_dir)
    ds_count = len(grasp_instructions_files)
    split = int(np.floor(0.8 * ds_count))

    if args.shuffle: 
        np.random.seed(42) # TODO-TRI: Replace 42 with a seed value
        np.random.shuffle(grasp_instructions_files)

    train_files, val_files = grasp_instructions_files[:split], grasp_instructions_files[split:]

    print('Training size: {}'.format(len(train_files)))
    print('Validation size: {}'.format(len(val_files)))
    
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        os.mkdir(train_grasp_instructions_dir)
        os.mkdir(train_grasp_label_dir)
        os.mkdir(train_images_dir)

        # Copy files to train directory
        for file in train_files:
            shutil.copy(os.path.join(grasp_instructions_dir, file), train_grasp_instructions_dir)
            shutil.copy(os.path.join(grasp_label_dir, file.replace('.pkl', '.pt')), train_grasp_label_dir)
            shutil.copy(os.path.join(images_dir, re.sub(r"_\d{1}_\d{1}\.pkl", ".jpg", file)), train_images_dir)

    if not os.path.exists(val_path):
        os.mkdir(val_path)
        os.mkdir(val_grasp_instructions_dir)
        os.mkdir(val_grasp_label_dir)
        os.mkdir(val_images_dir)

        # Copy files to val directory
        for file in val_files:
            shutil.copy(os.path.join(grasp_instructions_dir, file), val_grasp_instructions_dir)
            shutil.copy(os.path.join(grasp_label_dir, file.replace('.pkl', '.pt')), val_grasp_label_dir)
            shutil.copy(os.path.join(images_dir, re.sub(r"_\d{1}_\d{1}\.pkl", ".jpg", file)), val_images_dir)
