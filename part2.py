import torch
import matplotlib.pyplot as plt

from part2_utils import typesData

# Generates torsion mask. True iff a residue has meaningful torsion angles.
from my_part2 import get_torsion_mask

# Generates AA-type specific mask. True iff a residue has meaningful torsion angles and the specified type.
from my_part2 import get_type_mask

# Calculate phi/psi torsion angles and stores them in type specific data structures
from my_part2 import calculate_ramachandran_maps


def plot(types_data, aa_type, index):
    ax = plt.subplot(2, 2, index)
    type_data = types_data.types_data[aa_type]
    phi = type_data.ramachandran['phi']
    psi = type_data.ramachandran['psi']
    plt.scatter(phi, psi, s=1)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel('$\Phi$')  # Tex style
    plt.ylabel('$\Psi$')
    plt.title(type_data.name)
    ax.set_aspect('equal', adjustable='box')
    plt.xticks([-180, -90, 0, 90, 180])
    plt.yticks([-180, -90, 0, 90, 180])


def main():
    amino_acid_data = typesData()

    n_coordinates = torch.load(".\\refinementSampleData\\CoordNNative.pt")
    ca_coordinates = torch.load(".\\refinementSampleData\\CoordCaNative.pt")
    c_coordinates = torch.load(".\\refinementSampleData\\CoordCNative.pt")
    sequences = torch.load(".\\refinementSampleData\\sequences.pt")
    masks = torch.load(".\\refinementSampleData\\nativemask.pt")

    sequence0 = sequences[0]
    mask0 = masks[0] == 1
    # Task 1
    torsion_mask = get_torsion_mask(mask0)
    gly_torsion_mask = get_type_mask(torsion_mask, sequence0, 'G')
    f = open('part2out.txt', 'w')
    f.write(str(mask0) + "\n" + str(torsion_mask) + "\n" + str(gly_torsion_mask))
    f.close()

    # Task 2
    dict_phi_angels, dict_psi_angels = calculate_ramachandran_maps(amino_acid_data, sequences, masks, n_coordinates, ca_coordinates, c_coordinates)

    fig, axs = plt.subplots(1, 4)
    index = 0
    for amino_acid_name in amino_acid_data.types_data:
        if amino_acid_name == 'G' or amino_acid_name == 'W' or amino_acid_name == 'P' or amino_acid_name == 'A':
            axs[index].set_title(amino_acid_data.types_data[amino_acid_name].name)
            axs[index].set_xlim([-180, 180])
            axs[index].set_ylim([-180, 180])
            axs[index].set_ylabel(u'\u03A8')
            axs[index].set_xlabel(u'\u03A6')
            axs[index].scatter(dict_phi_angels[amino_acid_name].tolist(), dict_psi_angels[amino_acid_name].tolist())
            axs[index].set_aspect('equal', 'box')
            index += 1

    plt.savefig('results.png')
    plt.show()


if __name__ == '__main__':
    main()
    # Replace with the relevant path.
