import os
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np


def part1():
    ramachandran(os.getcwd() + "\\refinementSampleData", 11)
    calculateDistanceAndContactMatrix(
        os.getcwd() + "\\refinementSampleData", 11)  # Calculated Matrix Distances for a certain protein


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


def get_id(path, index):
    ids = torch.load(path + "/ids.pt")
    id = ids[index][0:8]  # cutting string.
    return id


def vector_dm(v):
    vv = v @ v.t()
    vv2 = 2 * vv
    vv_diagonal = vv.diag().unsqueeze(0)
    vv_diagonal2 = vv_diagonal + vv_diagonal.t()
    dm = (vv_diagonal2 - vv2).sqrt()
    return dm


def calculateDistanceAndContactMatrix(path, index):
    # We chose protien No.126
    id = get_id(path, index)
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    pro = ca_coordinates[index]
    ##
    Xpro = pro[:, 0]  # filtering only x cooardinates .
    XproFiltered = Xpro < 99999  # tensor of TRUE AND FALSE. TRUE WHERE coordinate is smaller than 99999 and false otherwise.
    sizepro = pro.shape[0]
    onezero_vector = torch.zeros(sizepro,
                                 1)  # creating a one(true) zero(false) vector accordingly. initialize it with zeros.
    onezero_vector[XproFiltered] = 1
    distance_map = onezero_vector @ onezero_vector.t()  # creating a matrix tenor instead of a vector.
    distance_map_ignoring_999 = vector_dm(pro)
    onezero_matrixFiltered = distance_map == 1
    # in every spot where there was 1 in the distance map , we replace it with the actual value stored in distance map ignoring 999.
    distance_map[onezero_matrixFiltered] = distance_map_ignoring_999[onezero_matrixFiltered]
    ##
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    fig.suptitle(id)
    ax2.set_title("Contact Map")
    contact_map = distance_map[:] / 100
    contact_map = contact_map[:] <= 10
    ax2.imshow(contact_map, cmap=matplotlib.cm.get_cmap("viridis"), interpolation="antialiased", origin="upper")
    ax1.set_title("Distance Map")
    ax1.imshow(distance_map, cmap=matplotlib.cm.get_cmap("viridis"), interpolation="antialiased", origin="upper")
    ram = ramachandran(path, index)
    ax3.set_title("Ramachandran")
    ax3.set_xlim([-180, 180])
    ax3.set_ylim([-180, 180])
    ax3.scatter(ram[0], ram[1])
    ax3.axvline(c='grey', lw=1.5)
    ax3.axhline(c='grey', lw=1.5)
    ax3.set_aspect('equal', 'box')
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
    mask_i = ca_coordinates_i[:, 0] < 9999
    mask_i = mask_i[1:-1]  # excluding the first and the last.
    n_coordinates_i = n_coordinates_i[:] / 100
    ca_coordinates_i = ca_coordinates_i[:] / 100
    c_coordinates_i = c_coordinates_i[:] / 100
    c0_phi = c_coordinates_i[0:-2, :]  # from 0 to n-2 included.
    # excluding the first and the last . only for n-2 amino acids in the middle.
    n_phi = n_coordinates_i[1:-1, :]
    ca_phi = ca_coordinates_i[1:-1, :]
    c_phi = c_coordinates_i[1:-1, :]
    phi, cosPhi, sinPhi = torsionAngle(c0_phi.t(), n_phi.t(), ca_phi.t(), c_phi.t())
    n_psi = n_coordinates_i[1:-1, :]
    ca_psi = ca_coordinates_i[1:-1, :]
    c_psi = c_coordinates_i[1:-1, :]
    n1_psi = n_coordinates_i[2:, :]  # last n-2 amino acids.
    psi, cosPsi, sinPsi = torsionAngle(n_psi.t(), ca_psi.t(), c_psi.t(), n1_psi.t())
    # creating a new correct mask according to the invalid places. initialize it with true.
    mask_size = len(mask_i)
    mask_i_fixed = [True] * (mask_size)
    counter = 0
    for element in mask_i:
        # neighbor of invalid place -> angle is invalid. cases: "right and left"
        if not element and counter > 0:
            mask_i_fixed[counter - 1] = False
            mask_i_fixed[counter] = False
        if not element and counter < len(mask_i) - 1:
            mask_i_fixed[counter + 1] = False
            mask_i_fixed[counter] = False
        counter = counter + 1
    phi = phi[0]
    psi = psi[0]
    # filtering phi and psi.
    phi = phi[mask_i_fixed]
    psi = psi[mask_i_fixed]
    return phi, psi
