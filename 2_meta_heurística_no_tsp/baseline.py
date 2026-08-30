def nearest_neighbor(distance_matrix, start = 0):
    """Calculates a route for the TSP using the 
    Nearest Neighbor constructive heuristic."""
    num_cities = len(distance_matrix)

    visited = [False] * num_cities
    route = [start]
    visited[start] = True
    current = start

    for i in range(num_cities -1):
        nearest_city = None
        nearest_dist = float("inf")

        for city in range(num_cities):
            if not visited[city] and distance_matrix[current][city] < nearest_dist:
                nearest_dist = distance_matrix[current][city]
                nearest_city = city

        route.append(nearest_city)
        visited[nearest_city] = True
        current = nearest_city

    return route
