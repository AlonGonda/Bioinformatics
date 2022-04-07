from functions import a_b_distance, calculateDistanceMatrix, read_ca_coordinates, a_b_distance_with_grad


def main():
    # print("Example of a_b_distance")
    print(a_b_distance(10, 5))
    # print("Example of a_b_dm")
    # print(a_b_dm(10, 5))

    distanceMatrix = calculateDistanceMatrix(
        "C:/Users/user/Desktop/third year/Bioinformatics/refinementSampleData")  # Calculated Matrix Distances for a certain protein
    print("Example of loading protein Coordinates")
    read_ca_coordinates("C:/Users/user/Desktop/third year/Bioinformatics/refinementSampleData")
    print("Back to a_b_distance this time with gradient")
    print(a_b_distance_with_grad(10., 5.))


if __name__ == '__main__':
    main()
