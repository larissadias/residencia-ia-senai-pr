import math

def read_tsp(file_path):
    """Reads a TSPLIB formatted file to extract the problem's name and the 
    coordinates of its cities."""
    name = ""
    coordinates = []
    reading = False

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                reading = True
            elif line.startswith("EOF"):
                break
            elif reading and line:
                values = line.split()
                coordinates.append((float(values[1]), float(values[2])))

    return name, coordinates


def build_distance_matrix(coodinates):
    """Constructs a symmetric 2D distance matrix from a list of city coordinates. 
    It calculates the Euclidean distance between all pairs of cities and rounds 
    the result to the nearest integer, adhering to the standard TSPLIB EUC_2D 
    distance metric conventions."""
    size = len(coodinates)
    matrix = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(i + 1, size):
            x_diff = coodinates[i][0] - coodinates[j][0]
            y_diff = coodinates[i][1] - coodinates[j][1]
            distance = int(round(math.hypot(x_diff, y_diff)))
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix

def calculates_distance(route, distance_matrix):
    """Calculates the total distance (cost) of a given route for the Traveling 
    Salesperson Problem. The total distance includes the cost of traveling 
    between all consecutive cities in the list, plus the cost of returning 
    from the last city back to the starting city."""
    distance = 0

    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i +1) % len(route)]
        distance += distance_matrix[current_city][next_city]

    return distance

def read_opt(file_path):
    """Reads a TSPLIB formatted optimal tour file (.opt.tour) to extract the 
    sequence of cities representing the best-known optimal solution. 
    
    This function automatically converts the 1-based city indices used in the 
    TSPLIB format into the 0-based indices used internally by the algorithms."""
    tour = []
    reading = False

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("TOUR_SECTION"):
                reading = True
            elif reading and line:
                for value in line.split():
                    city = int(value)
                    if city == -1:
                        return tour
                    tour.append(city - 1)

    return tour