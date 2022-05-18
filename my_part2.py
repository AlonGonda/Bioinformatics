import part2_utils
import torch


def get_amino_acid_indexes(sequences, masks, types_data):
    dict_amino_acids_global = {x: [] for x in types_data}
    for j, protein in enumerate(sequences):
        mask = masks[j] == 1
        torsion_mask = get_torsion_mask(mask)
        dict_torsion_mask = {x: get_type_mask(torsion_mask, protein, x) for x in types_data}
        dict_amino_acids_current = {x: [] for x in types_data}
        for amino_acid_name in types_data:
            for index, amino_acid in enumerate(dict_torsion_mask[amino_acid_name]):
                if amino_acid:
                    dict_amino_acids_current[amino_acid_name].append(index)
            dict_amino_acids_global[amino_acid_name].append(dict_amino_acids_current[amino_acid_name])

    for i in range(0, len(sequences)):
        for key in dict_amino_acids_global.keys():
            dict_amino_acids_global[key][i] = torch.IntTensor(dict_amino_acids_global[key][i])

    return dict_amino_acids_global


def calculate_ramachandran_maps(amino_data, sequences, masks, n_coordinates, ca_coordinates, c_coordinates):
    types_data = amino_data.types_data
    dict_amino_acids_global = get_amino_acid_indexes(sequences, masks, types_data)

    dict_phi_angels = {x: torch.tensor([], dtype=torch.int64) for x in types_data}
    dict_psi_angels = {x: torch.tensor([], dtype=torch.int64) for x in types_data}

    for index in range(0, len(sequences)):
        n_coordinates_current = n_coordinates[index]
        ca_coordinates_current = ca_coordinates[index]
        c_coordinates_current = c_coordinates[index]
        phi, psi = part2_utils.get_ramachandran(n_coordinates_current[:] / 100, ca_coordinates_current[:] / 100, c_coordinates_current[:] / 100)
        for amino_acid_name in types_data:
            phi_current = torch.index_select(phi[0], 0, dict_amino_acids_global[amino_acid_name][index].to(torch.int64))
            psi_current = torch.index_select(psi[0], 0, dict_amino_acids_global[amino_acid_name][index].to(torch.int64))
            dict_phi_angels[amino_acid_name] = torch.cat((dict_phi_angels[amino_acid_name], phi_current), 0)
            dict_psi_angels[amino_acid_name] = torch.cat((dict_psi_angels[amino_acid_name], psi_current), 0)

    return dict_phi_angels, dict_psi_angels


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
