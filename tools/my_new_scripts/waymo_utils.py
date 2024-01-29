import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

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
    
def switch_pkl_files(segment_file, src_dir, dest_dir):
    """
    PVT-SSD is using pkl files in another format, but the npy files are the same. This function
    realizes switching the pkl files between OpenPCDet standard format and PVT-SSD specified format.
    """
    import shutil
    segments = read_segment_file(segment_file)
    
    for segment in tqdm(segments):
        pkl_src_file = os.path.join(src_dir, segment, segment + '.pkl')
        pkl_dest_file = os.path.join(dest_dir, segment, segment + '.pkl')
        
        shutil.copy2(pkl_src_file, pkl_dest_file)
    
    
