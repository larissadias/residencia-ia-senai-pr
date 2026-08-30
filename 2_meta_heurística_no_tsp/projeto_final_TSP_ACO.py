import random
import time
from read_tsp import calculates_distance

NUM_ANTS = 50
NUM_ITERATIONS = 1000000
TIME_LIMIT = 30.0
ALPHA = 1.0
BETA = 5
EVAPORATION_RATE = 0.1
INITIAL_PHEROMONE = 1.0


def ant_colony_optimization(distance_matrix, time_limit=TIME_LIMIT):
    """Executes the Ant Colony Optimization (ACO) algorithm to find an optimal or 
    near-optimal route for the Traveling Salesperson Problem (TSP) within a 
    specified time limit."""
    start_time = time.perf_counter()
    num_cities = len(distance_matrix)

    pheromone = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            row.append(INITIAL_PHEROMONE)
        pheromone.append(row)

    heuristic = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0.0)
            else:
                inv_dist = 1.0 / distance_matrix[i][j]
                row.append(inv_dist ** BETA) 
        heuristic.append(row)

    global_best_route = []
    global_best_cost = float('inf')
    best_iteration = 0
    iterations_done = 0
    history = []

    for iteration in range(1, NUM_ITERATIONS +1):
        colony_routes = []
        colony_costs = []

        for ant in range(NUM_ANTS):
            current_city_idx = random.randint(0, num_cities - 1)
            
            route_indices = [current_city_idx]
            unvisited = set(range(num_cities))
            unvisited.remove(current_city_idx)

            while unvisited:
                probabilities = []
                total_sum = 0.0

                for next_city_idx in unvisited:
                    # O expoente BETA ja foi aplicado ao montar a matriz
                    tau = pheromone[current_city_idx][next_city_idx] ** ALPHA 
                    eta = heuristic[current_city_idx][next_city_idx] 
                    prob_value = tau * eta
                    probabilities.append((next_city_idx, prob_value))
                    total_sum += prob_value

                if total_sum == 0.0:
                    next_city_idx = random.choice(list(unvisited))
                else:
                    cities, probs = zip(*probabilities)
                    next_city_idx = random.choices(cities, weights=probs, k=1)[0]

                route_indices.append(next_city_idx)
                unvisited.remove(next_city_idx)
                current_city_idx = next_city_idx


            ant_route = route_indices[:]
            ant_cost = calculates_distance(ant_route, distance_matrix)

            colony_routes.append(ant_route)
            colony_costs.append(ant_cost)

            if ant_cost < global_best_cost:
                global_best_cost = ant_cost
                global_best_route = ant_route[:]
                best_iteration = iteration

        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] = (1.0 - EVAPORATION_RATE) * pheromone[i][j]

        for ant_idx in range(NUM_ANTS):
            route = colony_routes[ant_idx]
            cost = colony_costs[ant_idx]
            deposit = 100.0 / cost

            for i in range(len(route)):
                current_city = route[i]
                next_city = route[(i + 1) % len(route)]
                pheromone[current_city][next_city] += deposit
                pheromone[next_city][current_city] += deposit

        history.append(global_best_cost)
        iterations_done = iteration

        if time.perf_counter() - start_time >= time_limit:
            break

    return global_best_route, global_best_cost, best_iteration, history, iterations_done