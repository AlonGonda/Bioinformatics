import RamachanDraw
import torch
import matplotlib.pyplot as plt
import numpy as np


# import PyPDB
def load_pt_file(path):
    model = torch.load(path)
    protein = model[126]
    PDB_parser(protein)


# parse list of atom to the format of pdb file to ramachan diagram
def PDB_parser(protein):
    f1 = open("myfile.pdb", "w")
    i = 0
    for atom in protein:
        atom = atom.numpy()
        j1 = "ATOM".ljust(6)  # atom#6s
        index = str(++i)
        j2 = index.rjust(5)  # aomnum#5d
        j3 = "C".center(4)  # atomname$#4s
        j4 = "PRO".ljust(3)  # resname#1s
        j5 = "A".rjust(1)  # Astring
        j6 = index.rjust(4)  # resnum
        j7 = str('%8.3f' % (float(atom[0]))).rjust(8)  # x
        j8 = str('%8.3f' % (float(atom[1]))).rjust(8)  # y
        j9 = str('%8.3f' % (float(atom[2]))).rjust(8)  # z
        j10 = str('%6.2f' % (1)).rjust(6)  # occ
        j11 = str('%6.2f' % (60)).ljust(6)  # temp
        j12 = "C".rjust(12)  # elname

        text = "{first}{second} {third} {fourth} {fifth}{sixth}    {seventh}{eighth}{ninth}{tenth}{eleventh}{twelvth}\n".format(
            first=j1, second=j2, third=j3, fourth=j4, fifth=j5, sixth=j6, seventh=j7, eighth=j8, ninth=j9, tenth=j10,
            eleventh=j11, twelvth=j12)
        f1.write(text)


# create ramachan plot
def generate_ramachan(path):
    pdb_file = path + "\\myfile.pdb"
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
    # We chose protien No.126
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    first = ca_coordinates[126]
    distances = []
    for aminoAcidI in first:
        current = []
        for aminoAcidJ in first:
            current.append(
                torch.pairwise_distance(aminoAcidI, aminoAcidJ))  # calculating the distance between two amino acids 3D.
        # print(current) # prints the distance matrix
        distances.append(current)

    distanceMatrixInt = []
    for i in distances:
        new_line = []
        for j in i:
            new_line.append(j.item())
        distanceMatrixInt.append(new_line)
    print(plt.matshow(distanceMatrixInt))


# Read coordinates of C-alpha
def read_ca_coordinates(path):
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    # What is it that we loaded
    # print(type(ca_coordinates))
    #  <class 'list'>
    #  What is the length of the list?
    # print(len(ca_coordinates))
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
    out = ((a - b) ** 2).sqrt()
    out.backward()
    return out.item(), a.grad, b.grad


def torsionAngle(V1, V2, V3, V4):
    # V in 3xN
    A = V2 - V1
    B = V3 - V2
    C = V4 - V3

    Bsq = torch.relu(torch.sum(B * B, dim=0, keepdim=True))
    AC = torch.sum(A * C, dim=0, keepdim=True)
    AB = torch.sum(A * B, dim=0, keepdim=True)
    BC = torch.sum(B * C, dim=0, keepdim=True)
    x = -torch.sum(Bsq * AC, dim=0, keepdim=True) + torch.sum(AB * BC, dim=0, keepdim=True)

    absB = torch.sqrt(Bsq).sum(dim=0, keepdim=True)
    BxC = torch.cross(B, C)
    y = torch.sum((absB * A) * BxC, dim=0, keepdim=True)

    cosTheta = x / torch.sqrt(x ** 2 + y ** 2 + 1e-3)
    sinTheta = y / torch.sqrt(x ** 2 + y ** 2 + 1e-3)
    theta = torch.arccos(cosTheta)
    theta = theta * torch.sign(y)
    return 180 * theta / torch.pi, cosTheta, sinTheta
