import os

from matplotlib import pyplot as plt

import part2_utils
import torch


def get_amino_acid_indexes(sequences):
    masks = torch.load(".\\refinementSampleData\\nativemask.pt")
    dict_amino_acids_global = {}
    types_data = part2_utils.typesData().types_data
    for amino_acid_name in types_data:
        dict_amino_acids_global[amino_acid_name] = []
    j = 0
    for protein in sequences:
        mask = masks[j] == 1
        torsion_mask = get_torsion_mask(mask)
        dict_torsion_mask = {}
        dict_amino_acids_current = {}
        for amino_acid_name in types_data:
            dict_torsion_mask[amino_acid_name] = get_type_mask(torsion_mask, protein, amino_acid_name)
            dict_amino_acids_current[amino_acid_name] = []
            index = 0
            for amino_acid in dict_torsion_mask[amino_acid_name]:
                if amino_acid:
                    dict_amino_acids_current[amino_acid_name].append(index)
                index += 1
            dict_amino_acids_global[amino_acid_name].append(dict_amino_acids_current[amino_acid_name])
        j += 1

    for i in range(0, len(sequences)):
        for key in dict_amino_acids_global.keys():
            dict_amino_acids_global[key][i] = torch.IntTensor(dict_amino_acids_global[key][i])

    return dict_amino_acids_global


def calculate_ramachandran_maps():
    n_coordinates = torch.load(".\\refinementSampleData\\CoordNNative.pt")
    ca_coordinates = torch.load(".\\refinementSampleData\\CoordCaNative.pt")
    c_coordinates = torch.load(".\\refinementSampleData\\CoordCNative.pt")
    sequences = torch.load(".\\refinementSampleData\\sequences.pt")

    dict_amino_acids_global = get_amino_acid_indexes(sequences)
    types_data = part2_utils.typesData().types_data

    dict_phi_angels = {}
    dict_psi_angels = {}
    for amino_acid_name in types_data:
        dict_phi_angels[amino_acid_name] = torch.tensor([], dtype=torch.float32)
        dict_psi_angels[amino_acid_name] = torch.tensor([], dtype=torch.float32)

    for index in range(0, len(sequences)):
        n_coordinates_dict = {}
        ca_coordinates_dict = {}
        c_coordinates_dict = {}
        phi = {}
        psi = {}
        for amino_acid_name in types_data:
            n_coordinates_dict[amino_acid_name] = torch.index_select(n_coordinates[index], 0,
                                                                     dict_amino_acids_global['A'][index].to(
                                                                         torch.int64))
            ca_coordinates_dict[amino_acid_name] = torch.index_select(ca_coordinates[index], 0,
                                                                      dict_amino_acids_global['A'][index].to(
                                                                          torch.int64))
            c_coordinates_dict[amino_acid_name] = torch.index_select(c_coordinates[index], 0,
                                                                     dict_amino_acids_global['A'][index].to(
                                                                         torch.int64))

            phi[amino_acid_name], psi[amino_acid_name] = part2_utils.get_ramachandran(
                n_coordinates_dict[amino_acid_name][:],
                ca_coordinates_dict[amino_acid_name][:],
                c_coordinates_dict[amino_acid_name][:])

            dict_phi_angels[amino_acid_name] = torch.cat((dict_phi_angels[amino_acid_name], phi[amino_acid_name]), 1)
            dict_psi_angels[amino_acid_name] = torch.cat((dict_psi_angels[amino_acid_name], psi[amino_acid_name]), 1)

    fig, axs = plt.subplots(1, len(types_data))
    index = 0
    for amino_acid_name in types_data:
        axs[index].set_title(types_data[amino_acid_name].name)
        axs[index].set_xlim([-180, 180])
        axs[index].set_ylim([-180, 180])
        axs[index].set_ylabel(u'\u03A8')
        axs[index].set_xlabel(u'\u03A6')
        axs[index].scatter(dict_phi_angels['A'].tolist(), dict_psi_angels['A'].tolist())
        axs[index].set_aspect('equal', 'box')
        index += 1

    plt.show()


def get_type_mask(mask, seq, type):
    mask_type = [False] * len(mask)
    dictionary_types = enumerate(seq)
    for index, value in dictionary_types:
        if value == type:
            mask_type[index] = True
    return torch.as_tensor(mask_type)


def get_torsion_mask(mask):
    torsion_mask = [True] * len(mask)
    torsion_mask[0] = False  # no phi angle
    torsion_mask[-1] = False  # no psi angle
    dictionary_mask = enumerate(mask)
    for index, value in dictionary_mask:
        if value == False and index >= 1:
            torsion_mask[index - 1] = False
        if value == False and index <= len(mask) - 2:
            torsion_mask[index + 1] = False
    return torch.as_tensor(torsion_mask)
