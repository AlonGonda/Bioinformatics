import torch
import matplotlib.pyplot as plt
import numpy as np
import RamachanDraw


def generate_Ramachan():
    pdb_file = "C:/Users/user/Desktop/third year/Bioinformatics/refinementSampleData/CoordCaNative.pdb"
    RamachanDraw.plot(pdb_file, cmap='viridis', alpha=0.75, dpi=100, save=True, show=False, out='plot.png')


# Simple distance

def a_b_distance(a, b):
    a = torch.tensor(a)
    print(a)
    b = torch.tensor(b)
    out = ((a - b) ** 2).sqrt()
    return out.item()


# Distance map of two variables
def a_b_dm(a, b):
    ab = torch.FloatTensor(2, 1)  # A 2 X 1 tensor, column vector
    ab[0, 0] = a
    ab[1, 0] = b
    ab = ab @ ab.t()  # column vector times row vector = 3 X2 square matrix
    ab2 = 2 * ab  # Scalar multiplication
    ab_diagonal = ab.diag().unsqueeze(0)  # Unsqueeze adds dimension to the tensor. Here from 2 to 1 X 2
    ab_diagonal2 = ab_diagonal + ab_diagonal.t()
    dm = (ab_diagonal2 - ab2).sqrt()
    return dm


def calculateDistanceMatrix(path):
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    first = ca_coordinates[126]
    distances = []
    for aminoAcidI in first:
        current = []
        for aminoAcidJ in first:
            current.append(torch.pairwise_distance(aminoAcidI, aminoAcidJ)) #calculating the distance between two amino acids 3D.
        print(current)
        distances.append(current)
    return distances



# Read coordinates of C-alpha
def read_ca_coordinates(path):
    ca_coordinates = torch.load(path+"/CoordCaNative.pt")
    # What is it that we loaded
    #print(type(ca_coordinates))
    #  <class 'list'>
    #  What is the length of the list?
    #print(len(ca_coordinates))
    #  153
    #  Lets look at the first element of the list
    first = ca_coordinates[126]

    #  What is it?
    print(type(first))
    #  <class 'torch.Tensor'>
    #  Of what dimensions?
    print(first.shape)
    #  torch.Size([215, 3])  That is the protein has 215 C-alpha atoms (rows), each with X, Y, and Z coordinates
    plt.figure()
    ax = plt.axes(projection='3d')
    indices = first[:, 1] < 999
    x = np.array(first[indices, 0])
    y = np.array(first[indices, 1])
    z = np.array(first[indices, 2])
    ax.plot3D(x, y, z, 'gray')
    plt.show()


def a_b_distance_with_grad(a, b):
    a = torch.tensor([a], requires_grad=True)
    b = torch.tensor([b], requires_grad=True)
    out = ((a - b) ** 2 ).sqrt()
    out.backward()
    return out.item(), a.grad, b.grad
