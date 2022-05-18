import torch


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


def get_ramachandran(n_coordinates, ca_coordinates, c_coordinates):
    c0_phi = c_coordinates[0:-2, :]
    n_phi = n_coordinates[1:-1, :]
    ca_phi = ca_coordinates[1:-1, :]
    c_phi = c_coordinates[1:-1, :]
    phi_tmp, cos_phi, sin_phi = torsionAngle(c0_phi.t(), n_phi.t(), ca_phi.t(), c_phi.t())
    phi = torch.zeros(1, phi_tmp.shape[1] + 2)  # Why?
    phi[0, 1:-1] = phi_tmp[0, :]
    n_psi = n_coordinates[1:-1, :]
    ca_psi = ca_coordinates[1:-1, :]
    c_psi = c_coordinates[1:-1, :]
    n1_psi = n_coordinates[2:, :]
    psi_tmp, cos_psi, sin_psi = torsionAngle(n_psi.t(), ca_psi.t(), c_psi.t(), n1_psi.t())
    psi = torch.zeros(phi.shape)
    psi[0, 1:-1] = psi_tmp[0, :]
    return phi, psi


class typeData:
    def __init__(self, name, name1, name3):
        self.name = name
        self.name1 = name1
        self.name3 = name3
        self.ramachandran = {'phi': torch.tensor([], dtype=torch.float32), 'psi': torch.tensor([], dtype=torch.float32)}


class typesData:
    def __init__(self):
        self.types_data = {
            'A': typeData('Alanine', 'Ala', 'A'),
            'C': typeData('Cysteine', 'Cys', 'A'),
            'D': typeData('Aspartic Acid', 'Asp', 'D'),
            'E': typeData('Glutamic Acid', 'Glu', 'E'),
            'F': typeData('Phenylalanine', 'Phe', 'F'),
            'G': typeData('Glycine', 'Gly', 'G'),
            'H': typeData('Histidine', 'His', 'H'),
            'I': typeData('Isoleucine', 'Ile', 'I'),
            'K': typeData('Lysine', 'Lys', 'K'),
            'L': typeData('Leucine', 'Leu', 'L'),
            'M': typeData('Methionine', 'Met', 'M'),
            'N': typeData('Asparagine', 'Asn', 'N'),
            'P': typeData('Proline', 'Pro', 'P'),
            'Q': typeData('Glutamione', 'Gln', 'Q'),
            'R': typeData('Arginine', 'Arg', 'R'),
            'S': typeData('Serine', 'Ser', 'S'),
            'T': typeData('Threonine', 'Thr', 'T'),
            'V': typeData('Valine', 'Val', 'V'),
            'W': typeData('Tryptophan', 'Trp', 'W'),
            'Y': typeData('Tyrosine', 'Tyr', 'Y')
        }


def get_mask(mask_string):
    return torch.tensor([c == '+' for c in mask_string])
