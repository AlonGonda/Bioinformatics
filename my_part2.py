import os

from matplotlib import pyplot as plt

import part2_utils
import torch


def get_amino_acid_indexes(sequences):
    dict_amino_acids_global = {'A': [], 'W': [], 'G': [], 'P': []}
    for protein in sequences:

        dict_amino_acids_current = {'A': [], 'W': [], 'G': [], 'P': []}

        index = 0
        for amino_acid in protein:
            if amino_acid == "A":
                dict_amino_acids_current['A'].append(index)
            elif amino_acid == "W":
                dict_amino_acids_current['W'].append(index)
            elif amino_acid == "G":
                dict_amino_acids_current['G'].append(index)
            elif amino_acid == "P":
                dict_amino_acids_current['P'].append(index)
            index += 1
        dict_amino_acids_global['A'].append(dict_amino_acids_current['A'])
        dict_amino_acids_global['W'].append(dict_amino_acids_current['W'])
        dict_amino_acids_global['G'].append(dict_amino_acids_current['G'])
        dict_amino_acids_global['P'].append(dict_amino_acids_current['P'])

    for i in range(0, len(sequences)):
        dict_amino_acids_global['A'][i] = torch.FloatTensor(dict_amino_acids_global['A'][i])
        dict_amino_acids_global['W'][i] = torch.FloatTensor(dict_amino_acids_global['W'][i])
        dict_amino_acids_global['G'][i] = torch.FloatTensor(dict_amino_acids_global['G'][i])
        dict_amino_acids_global['P'][i] = torch.FloatTensor(dict_amino_acids_global['P'][i])
    return dict_amino_acids_global


def calculate_ramachandran_maps():
    n_coordinates = torch.load(os.getcwd() + "\\refinementSampleData\\CoordNNative.pt")
    ca_coordinates = torch.load(os.getcwd() + "\\refinementSampleData\\CoordCaNative.pt")
    c_coordinates = torch.load(os.getcwd() + "\\refinementSampleData\\CoordCNative.pt")
    sequences = torch.load(os.getcwd() + "\\refinementSampleData\\sequences.pt")

    dict_amino_acids_global = get_amino_acid_indexes(sequences)
    dict_phi_angels = {'A': torch.tensor([], dtype=torch.float32), 'W': torch.tensor([], dtype=torch.float32),
                       'G': torch.tensor([], dtype=torch.float32), 'P': torch.tensor([], dtype=torch.float32)}
    dict_psi_angels = {'A': torch.tensor([], dtype=torch.float32), 'W': torch.tensor([], dtype=torch.float32),
                       'G': torch.tensor([], dtype=torch.float32), 'P': torch.tensor([], dtype=torch.float32)}
    for index in range(0, len(sequences)):
        n_coordinates_A = torch.index_select(torch.FloatTensor(n_coordinates[index]), 0,
                                             dict_amino_acids_global['A'][index].to(torch.int64))
        ca_coordinates_A = torch.index_select(torch.FloatTensor(ca_coordinates[index]), 0,
                                              dict_amino_acids_global['A'][index].to(torch.int64))
        c_coordinates_A = torch.index_select(torch.FloatTensor(c_coordinates[index]), 0,
                                             dict_amino_acids_global['A'][index].to(torch.int64))
        n_coordinates_P = torch.index_select(torch.FloatTensor(n_coordinates[index]), 0,
                                             dict_amino_acids_global['P'][index].to(torch.int64))
        ca_coordinates_P = torch.index_select(torch.FloatTensor(ca_coordinates[index]), 0,
                                              dict_amino_acids_global['P'][index].to(torch.int64))
        c_coordinates_P = torch.index_select(torch.FloatTensor(c_coordinates[index]), 0,
                                             dict_amino_acids_global['P'][index].to(torch.int64))
        n_coordinates_W = torch.index_select(torch.FloatTensor(n_coordinates[index]), 0,
                                             dict_amino_acids_global['W'][index].to(torch.int64))
        ca_coordinates_W = torch.index_select(torch.FloatTensor(ca_coordinates[index]), 0,
                                              dict_amino_acids_global['W'][index].to(torch.int64))
        c_coordinates_W = torch.index_select(torch.FloatTensor(c_coordinates[index]), 0,
                                             dict_amino_acids_global['W'][index].to(torch.int64))
        n_coordinates_G = torch.index_select(torch.FloatTensor(n_coordinates[index]), 0,
                                             dict_amino_acids_global['G'][index].to(torch.int64))
        ca_coordinates_G = torch.index_select(torch.FloatTensor(ca_coordinates[index]), 0,
                                              dict_amino_acids_global['G'][index].to(torch.int64))
        c_coordinates_G = torch.index_select(torch.FloatTensor(c_coordinates[index]), 0,
                                             dict_amino_acids_global['G'][index].to(torch.int64))

        phiA, psiA = part2_utils.get_ramachandran(n_coordinates_A, ca_coordinates_A, c_coordinates_A)
        phiP, psiP = part2_utils.get_ramachandran(n_coordinates_P, ca_coordinates_P, c_coordinates_P)
        phiW, psiW = part2_utils.get_ramachandran(n_coordinates_W, ca_coordinates_W, c_coordinates_W)
        phiG, psiG = part2_utils.get_ramachandran(n_coordinates_G, ca_coordinates_G, c_coordinates_G)

        dict_phi_angels['A'] = torch.cat((dict_phi_angels['A'], phiA), 1)
        dict_phi_angels['P'] = torch.cat((dict_phi_angels['P'], phiP), 1)
        dict_phi_angels['W'] = torch.cat((dict_phi_angels['W'], phiW), 1)
        dict_phi_angels['G'] = torch.cat((dict_phi_angels['G'], phiG), 1)

        dict_psi_angels['A'] = torch.cat((dict_psi_angels['A'], psiA), 1)
        dict_psi_angels['P'] = torch.cat((dict_psi_angels['P'], psiP), 1)
        dict_psi_angels['W'] = torch.cat((dict_psi_angels['W'], psiW), 1)
        dict_psi_angels['G'] = torch.cat((dict_psi_angels['G'], psiG), 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].set_title(part2_utils.typesData().types_data['A'].name)
    axs[0, 0].set_xlim([-180, 180])
    axs[0, 0].set_ylim([-180, 180])
    axs[0, 0].set_ylabel(u'\u03A8')
    axs[0, 0].set_xlabel(u'\u03A6')
    axs[0, 0].scatter(dict_phi_angels['A'].tolist(), dict_psi_angels['A'].tolist())
    axs[0, 0].set_aspect('equal', 'box')

    axs[1, 1].set_title(part2_utils.typesData().types_data['P'].name)
    axs[1, 1].set_xlim([-180, 180])
    axs[1, 1].set_ylim([-180, 180])
    axs[1, 1].set_ylabel(u'\u03A8')
    axs[1, 1].set_xlabel(u'\u03A6')
    axs[1, 1].scatter(dict_phi_angels['P'].tolist(), dict_psi_angels['P'].tolist())
    axs[1, 1].set_aspect('equal', 'box')

    axs[0, 1].set_title(part2_utils.typesData().types_data['W'].name)
    axs[0, 1].set_xlim([-180, 180])
    axs[0, 1].set_ylim([-180, 180])
    axs[0, 1].set_ylabel(u'\u03A8')
    axs[0, 1].set_xlabel(u'\u03A6')
    axs[0, 1].scatter(dict_phi_angels['W'].tolist(), dict_psi_angels['W'].tolist())
    axs[0, 1].set_aspect('equal', 'box')

    axs[1, 0].set_title(part2_utils.typesData().types_data['G'].name)
    axs[1, 0].set_xlim([-180, 180])
    axs[1, 0].set_ylim([-180, 180])
    axs[1, 0].set_ylabel(u'\u03A8')
    axs[1, 0].set_xlabel(u'\u03A6')
    axs[1, 0].scatter(dict_phi_angels['G'].tolist(), dict_psi_angels['G'].tolist())
    axs[1, 0].set_aspect('equal', 'box')

    plt.show()


def get_type_mask():
    return None


def get_torsion_mask():
    return None
