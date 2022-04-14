import os
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np


def part1():
    ramachandran(os.getcwd() + "\\refinementSampleData", 0)
    calculateDistanceAndContactMatrix(
        os.getcwd() + "\\refinementSampleData", 0)  # Calculated Matrix Distances for a certain protein


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


def calculateDistanceAndContactMatrix(path, index):
    # We chose protien No.126
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    first = ca_coordinates[index]
    distances = []
    for aminoAcidI in first:
        current = []
        for aminoAcidJ in first:
            current.append(
                torch.pairwise_distance(aminoAcidI,
                                        aminoAcidJ).numpy())  # calculating the distance between two amino acids 3D.
        distances.append(current)

    plt.matshow(distances)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Contact Map")

    contact_map = []
    for i in distances:
        new_line = []
        for j in i:
            toEnter = j.item() / 100
            if toEnter > 10:
                new_line.append(0)
            else:
                new_line.append(1)
        contact_map.append(new_line)

    axc = ax.imshow(contact_map, cmap=matplotlib.cm.get_cmap("viridis"), interpolation="antialiased", origin="upper")
    plt.show()


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


def ramachandran(path, index):
    n_coordinates = torch.load(path + "\\CoordNNative.pt")
    ca_coordinates = torch.load(path + "\\CoordCaNative.pt")
    c_coordinates = torch.load(path + "\\CoordCNative.pt")

    n_coordinates_i = n_coordinates[index]
    ca_coordinates_i = ca_coordinates[index]
    c_coordinates_i = c_coordinates[index]
    mask_i = n_coordinates_i[:, 0] < 9999

    n_coordinates_i = n_coordinates_i[mask_i, :] / 100
    ca_coordinates_i = ca_coordinates_i[mask_i, :] / 100
    c_coordinates_i = c_coordinates_i[mask_i, :] / 100

    c0_phi = c_coordinates_i[0:-1, :]
    n_phi = n_coordinates_i[1:, :]
    ca_phi = ca_coordinates_i[1:, :]
    c_phi = c_coordinates_i[1:, :]
    phi, cosPhi, sinPhi = torsionAngle(c0_phi.t(), n_phi.t(), ca_phi.t(), c_phi.t())
    n_psi = n_coordinates_i[0:-1, :]
    ca_psi = ca_coordinates_i[0:-1, :]
    c_psi = c_coordinates_i[0:-1, :]
    n1_psi = n_coordinates_i[1:, :]
    psi, cosPsi, sinPsi = torsionAngle(n_psi.t(), ca_psi.t(), c_psi.t(), n1_psi.t())

    fig, (axr) = plt.subplots(ncols=1, figsize=(4.2, 4.2))
    axr.set_title('Ramachandran map')

    plt.scatter(phi, psi)
    axr.axvline(c='grey', lw=1.5)
    axr.axhline(c='grey', lw=1.5)
    axr.set_aspect('equal', 'box')
