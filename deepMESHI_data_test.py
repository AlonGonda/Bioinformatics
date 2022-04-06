import os
import sys
import torch
from meshiData import MeshiDataset
from torch.utils.data import DataLoader as torchDataLoader
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_workers = 0
if sys.gettrace() is not None:
    num_workers = 0
print_files = False
if "s" in sys.argv:
    print_files = True

path = 'C:\\Users\\keasar\\Documents\\Work\\teaching\\mini_project\\deepMESHI2022\\data\\refinementSampleData'

decoyTrainData = MeshiDataset(path)
decoyTrainLoader = torchDataLoader(decoyTrainData, batch_size=1, shuffle=True,
                                   num_workers=num_workers)

for datum in decoyTrainLoader:
    id, msk, Coords, nativemask, CoordsNative, gdtt, iddt, embedding = datum

    print(datum)

