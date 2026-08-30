import random
import sys
import time
from read_tsp import read_tsp, build_distance_matrix, calculates_distance


POPULATION_SIZE = 100
STOPPING_CONDITION = 1000000
TIME_LIMIT = 30.0
MUTATION_RATE = 0.2
CROSSOVER_TYPE = "ox"  # Options: "pmx" ou "ox"
MUTATION_TYPE = "inversion" # Options: "swap" ou "inversion"

def fitness(route, distance_matrix):
    """Calculates the fitness."""
    dist = calculates_distance(route, distance_matrix)
    return 1.0 / dist

def tournament_selection(population, distance_matrix):
    """Select 3 individuals randomly and returns 
    the one with the best fitness"""
    tournament = random.sample(population, 3)
    winner = max(tournament, key=lambda route: fitness(route, distance_matrix))
    return winner

def pmx_crossover(parent1, parent2):
    "Applies PMX crossover"
    size = len(parent1)
    child1, child2 = [None]*size, [None]*size
  
    cut1, cut2 = sorted(random.sample(range(size), 2))
        
    child1[cut1:cut2+1] = parent2[cut1:cut2+1]
    child2[cut1:cut2+1] = parent1[cut1:cut2+1]
        
    map1 = {parent2[i]: parent1[i] for i in range(cut1, cut2+1)}
    map2 = {parent1[i]: parent2[i] for i in range(cut1, cut2+1)}
      
    for i in list(range(0, cut1)) + list(range(cut2+1, size)):
        val1 = parent1[i]
        while val1 in map1:
            val1 = map1[val1]
        child1[i] = val1
            
        val2 = parent2[i]
        while val2 in map2:
            val2 = map2[val2]
        child2[i] = val2

    return child1, child2

def build_ox_child(parent1, parent2):
    """Creates a child permutation from two parent permutations using the Order 
    Crossover (OX) operator"""
    size = len(parent1)
    child = [None]*size

    cut1, cut2 = sorted(random.sample(range(size), 2))
    segment = set()
    for i in range(cut1, cut2+1):
        child[i] = parent1[i]
        segment.add(parent1[i])

    position = (cut2 + 1) % size
    for i in range(size):
        city = parent2[(cut2 + 1 + i) % size]
        if city not in segment:
            child[position] = city
            position = (position + 1) % size

    return child

def ox_crossover(parent1, parent2):
    """Executes the Order Crossover (OX) operator to generate a pair of offspring 
    from two parent permutations."""
    child1 = build_ox_child(parent1, parent2)
    child2 = build_ox_child(parent2, parent1)

    return child1, child2

def swap_mutation(route):
    """Applies the swap mutation operator to a route permutation."""
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]

    return route

def inversion_mutation(route):
    """Applies the inversion mutation operator to a route permutation."""
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    route[idx1:idx2+1] = reversed(route[idx1:idx2+1])

    return route

def crossover(parent1, parent2):
    """Routes the crossover operation to the selected strategy based on the 
    global configuration setting."""
    if CROSSOVER_TYPE == "pmx":
        return pmx_crossover(parent1, parent2)
    else:
        return ox_crossover(parent1, parent2)

def mutation(route):
    """Probabilistically applies a mutation operator to a route permutation based 
    on the globally defined mutation rate."""
    if random.random() < MUTATION_RATE:
        if MUTATION_TYPE == "swap":
            return swap_mutation(route)
        else:
            return inversion_mutation(route)
        
    return route


def genetic_algoritm(distance_matrix, time_limit=TIME_LIMIT):
    """Executes a Genetic Algorithm to find an optimal or near-optimal route for 
    the Traveling Salesperson Problem (TSP), strictly respecting a specified time limit."""
    start_time = time.perf_counter()
    num_cities = len(distance_matrix)
    population = []

    for i in range(POPULATION_SIZE):
        random_route = random.sample(range(num_cities), num_cities)
        population.append(random_route)

    history = []
    best_route = population[0]
    best_cost = calculates_distance(best_route, distance_matrix)
    best_generation = 0
    generations_done = 0
    
    for generation in range(1, STOPPING_CONDITION +1):
        population.sort(key=lambda route: fitness(route, distance_matrix), reverse=True)
        generation_best = population[0]
        generation_cost = calculates_distance(generation_best, distance_matrix)

        if generation_cost < best_cost:
            best_cost = generation_cost
            best_route = generation_best[:]
            best_generation = generation

        history.append(best_cost)

        # Elitism
        next_population = []
        next_population.append(generation_best)

        # Tournament
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, distance_matrix)
            parent2 = tournament_selection(population, distance_matrix)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1)
            child2 = mutation(child2)

            next_population.extend([child1, child2])

        population = next_population[:POPULATION_SIZE]
        generations_done = generation

        if time.perf_counter() - start_time >= time_limit:
            break

    
    return best_route, best_cost, best_generation, history, generations_done


def format_route(route):
    cities = [str(city + 1) for city in route]
    return " -> ".join(cities) + " -> " + cities[0]


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Digite o caminho do arquivo .tsp: ").strip()
    
    name, coordinates = read_tsp(file_path)
    distance_matrix = build_distance_matrix(coordinates)
    
    start_time = time.perf_counter()
    best_route, best_cost, best_generation, history, generations_done = genetic_algoritm(distance_matrix)
    execution_time = time.perf_counter() - start_time
    
    print(f"\nArquivo de Entrada: {name}.tsp")
    print(f"Operadores: {CROSSOVER_TYPE.upper()} + {MUTATION_TYPE}")
    print(f"Melhor Rota Encontrada: {format_route(best_route)}")
    print(f"Custo Total: {best_cost}")
    print(f"Tempo de Execução: {execution_time:.1f} segundos")
    print(f"Geracoes concluidas em {TIME_LIMIT:.0f}s: {generations_done}")
    print(f"Melhor solução encontrada na geração {best_generation}")
    print(f"Custo inicial: {history[0]} | Custo final: {history[-1]}\n")
    

if __name__ == "__main__":
    main()