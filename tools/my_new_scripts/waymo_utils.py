import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
import multiprocessing as mp

import re
import pickle
from pathlib import Path


def read_segment_file(segment_file):
    """
    Get segments list from the segment file (e.g. the val.txt or train.txt from OpenPCDet)
    
    Args:
        segment_file: the file path of the segment file.
        
    Returns:
        segments: a list of segments.
    """
    with open(segment_file, 'r') as f:
        segments = f.readlines()
        
    segments = list(map(lambda x: x.rstrip()[:-len('.tfrecord')] if x.rstrip().endswith('.tfrecord') else x.rstrip(), segments))
        
    return segments


def copy_pkl_file(src_dir, dest_dir, segment):
    os.makedirs(os.path.join(dest_dir, segment), exist_ok=True)
    src_pkl = os.path.join(src_dir, segment, "{}.pkl".format(segment))
    dest_pkl = os.path.join(dest_dir, segment, "{}.pkl".format(segment))
    os.system("cp {} {}".format(src_pkl, dest_pkl   ))

def wrapper_func(args):
    src_dir, dest_dir, segment = args
    copy_pkl_file(src_dir, dest_dir, segment)

def switch_pkl_files(segment_file, src_dir, dest_dir):
    """
    PVT-SSD is using pkl files in another format, but the npy files are the same. This function
    realizes switching the pkl files between OpenPCDet standard format and PVT-SSD specified format.
    """
    segments = read_segment_file(segment_file)
    
    with mp.Pool(8) as pool:
        tasks = [(src_dir, dest_dir, segment) for segment in segments]
        for _ in tqdm(pool.imap_unordered(wrapper_func, tasks), total=len(tasks), desc='Copying pkl_files'):
            pass

def read_last_section(filename, marker="OBJECT_TYPE"):
    lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        file.seek(0, 2)  # Move the cursor to the end of the file
        file_size = file.tell()
        block_size = 1024
        buffer = ''

        # Starting from the end of the file, seek backwards in block_size chunks
        for position in range(file_size, file_size-4*block_size, -block_size):
            start_pos = max(0, position - block_size)  # Ensure start_pos is not negative
            file.seek(start_pos)
            buffer = file.read(min(block_size, position)) + buffer

            # Check if the marker is in this chunk, and split into lines if it is
            if marker in buffer:
                lines = buffer.splitlines()

    # Extract relevant lines (reverse to process in the original order)
    relevant_lines = [line for line in reversed(lines) if line.startswith(marker)]
    return relevant_lines

def parse_line(line):
    pattern = r"OBJECT_TYPE_TYPE_([A-Z]+)_LEVEL_([0-9]+): \[mAP ([0-9.]+)\] \[mAPH ([0-9.]+)\]"
    match = re.match(pattern, line)
    if match:
        # Extract captured groups: object class, level, mAP, and mAPH
        object_class, level, map_value, maph_value = match.groups()
        return object_class, level, float(map_value), float(maph_value)
    else:
        # Return None or raise an error if the line doesn't match the pattern
        return None

def aggregate_data(filename, save_file, marker="OBJECT_TYPE"):
    results = defaultdict(dict)
    relevant_lines = read_last_section(filename, marker)
    directory_path = Path(save_file).parent
    os.makedirs(directory_path, exist_ok=True)
    
    for line in relevant_lines:
        object_class, level, map_value, maph_value = parse_line(line)
        results[object_class][f'LEVEL_{level}'] = {
            'mAP': map_value,
            'mAPH': maph_value
        }
    
    # print(results)
    with open(save_file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    