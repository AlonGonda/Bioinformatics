from multiprocessing import freeze_support

import torch
from torch.utils.data import Dataset
import numpy as np
import os
"""
import utils
from torch.autograd import grad
"""
import glob


class MeshiDataset(Dataset):
    """Dataset of refinements from AlphaFold and RosettaFold."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        protein_path = self.root_dir

        self.seq = torch.load(protein_path + '\\seq.pt')
        self.ids = torch.load(protein_path + '/ids.pt')

        self.coordN = torch.load(protein_path + '/CoordN.pt')
        self.coordAlpha = torch.load(protein_path + '/CoordAlpha.pt')
        self.coordC = torch.load(protein_path + '/CoordC.pt')
        self.coordBeta = torch.load(protein_path + '/CoordBeta.pt')

        self.nativemask = torch.load(protein_path + '/nativemask.pt')
        self.msk = torch.load(protein_path + '/mask.pt')

        self.gdtts = torch.load(protein_path + '/GDTTS.pt')
        self.iddts = torch.load(protein_path + '/IDDTS.pt')

        self.coordNNative = torch.load(protein_path + '/CoordNNative.pt')
        self.coordAlphaNative = torch.load(protein_path + '/CoordCaNative.pt')
        self.coordCNative = torch.load(protein_path + '/CoordCNative.pt')
        self.coordBetaNative = torch.load(protein_path + '/CoordCbNative.pt')

        self.embeddings = torch.load(protein_path + '/embeddings.pt')

    def __len__(self):
        return len(self.seq)

    def read_protein_data(self, i):
        # proteinPath is the path to the folder with all the .pt files of the proteins
        protein_path = self.root_dir
        seq = self.seq[i]
        ids = self.ids[i]

        coordN = self.coordN[i]
        coordAlpha = self.coordAlpha[i]
        coordC = self.coordC[i]
        coordBeta = self.coordBeta[i]

        nativemask = self.nativemask[i]
        msk = self.msk[i]

        gdtts = self.gdtts[i]
        iddts = self.iddts[i]

        coordNNative = self.coordNNative[i]
        coordAlphaNative = self.coordAlphaNative[i]
        coordCNative = self.coordCNative[i]
        coordBetaNative = self.coordBetaNative[i]

        embeddings = self.embeddings[i]

        return coordN, coordAlpha, coordC, coordBeta, seq, ids, msk, gdtts, iddts, embeddings, coordNNative, coordAlphaNative, coordCNative, coordBetaNative, nativemask

    def __getitem__(self, idx):
        coordN, coordAlpha, coordC, coordBeta, seq, id, msk, gdtt, \
        iddt, embedding, coordNNative, coordAlphaNative, \
        coordCNative, coordBetaNative, nativemask = self.read_protein_data(idx)
        scale = 1e-2

        msk = msk.type('torch.FloatTensor')
        nativemask = nativemask.type('torch.FloatTensor')

        X1 = coordAlpha.t()
        X2 = coordC.t()
        X3 = coordN.t()
        X4 = coordBeta.t()

        X1native = coordAlphaNative.t()
        X2native = coordCNative.t()
        X3native = coordNNative.t()
        X4native = coordBetaNative.t()

        CoordsNative = scale * torch.stack((X1native, X2native, X3native, X4native), dim=1) # 4x3xN CA|C|N|CB
        CoordsNative = CoordsNative.type('torch.FloatTensor')

        Coords = scale * torch.stack((X1, X2, X3, X4), dim=1)
        Coords = Coords.type('torch.FloatTensor')


        return id, msk, Coords, nativemask, CoordsNative, gdtt, iddt, embedding

