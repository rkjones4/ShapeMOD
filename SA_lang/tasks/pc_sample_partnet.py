import os, sys
import numpy as np
from tqdm import tqdm
import utils

category = sys.argv[1]
odir = f'pc_data/{category}'
PARTNET_PATH = PATH_TO_PARTNET # Replace with actual path

def loadPts(infile):
    pts = []
    with open(infile) as f:
        for line in f:
            ls = line[:-1].split()
            pts.append([float(l) for l in ls])

    pts = np.array(pts).astype('float32')
    offset = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts -= offset
    
    return pts

def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

train_inds = getInds(f'pc_data_splits/{category}/train.txt')
val_inds = getInds(f'pc_data_splits/{category}/val.txt')
test_inds = getInds(f'pc_data_splits/{category}/test.txt')

all_inds = list(train_inds.union(val_inds).union(test_inds))

os.system(f'mkdir {odir}')

for ind in tqdm(all_inds):
    pts = loadPts(f'{PARTNET_PATH}/{ind}/point_sample/pts-10000.txt')
    np.save(f'{odir}/{ind}.pts', pts)
