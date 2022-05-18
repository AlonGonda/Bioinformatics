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
    types_data = typesData()

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
    calculate_ramachandran_maps(types_data, sequences, masks, n_coordinates, ca_coordinates, c_coordinates)
    # plot(types_data, 'A', 1)
    # plot(types_data, 'W', 2)
    # plot(types_data, 'G', 3)
    # plot(types_data, 'P', 4)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
    # Replace with the relevant path.
