import random
import time
import matplotlib.pyplot as plt
import statistics
from read_tsp import read_tsp, build_distance_matrix, calculates_distance, read_opt
from baseline import nearest_neighbor
from projeto_final_TSP_GA import genetic_algoritm, POPULATION_SIZE, STOPPING_CONDITION, CROSSOVER_TYPE, MUTATION_TYPE
from projeto_final_TSP_ACO import ant_colony_optimization, NUM_ANTS


INSTANCES = [
    ("Leve",          "data/pr76.tsp",   "data/pr76.opt.tour"),
    ("Intermediária", "data/tsp225.tsp", "data/tsp225.opt.tour"),
    ("Pesada",        "data/pcb442.tsp", "data/pcb442.opt.tour"),
]


#INSTANCES = [
#    ("Leve", "data/pr76.tsp", "data/pr76.opt.tour"),
#]

NUM_SEEDS = 3
TIME_LIMIT = 30.0


def gap(cost,opt_cost):
    return 100.0 * (cost - opt_cost) / opt_cost

def summarize(runs, opt_cost):
    costs = [r["cost"] for r in runs]
    return {
        "runs": runs,
        "best_cost": min(costs),
        "mean_cost": statistics.mean(costs),
        "stdev_cost": statistics.stdev(costs) if len(costs) > 1 else 0.0,
        "worst_cost": max(costs),
        "mean_gap": gap(statistics.mean(costs), opt_cost),
        "mean_time": statistics.mean(r["time"] for r in runs),
        "mean_iterations": statistics.mean(r["iterations"] for r in runs),
        "median_run": sorted(runs, key=lambda r: r["cost"])[len(runs) // 2],
    }

def run_instances(label, tsp_path, opt_path):
    name, coordinates = read_tsp(tsp_path)
    distance_matrix = build_distance_matrix(coordinates)
    num_cities = len(coordinates)

    opt_route = read_opt(opt_path)
    opt_cost = calculates_distance(opt_route, distance_matrix)

    initial_time = time.perf_counter()
    nn_route = nearest_neighbor(distance_matrix)
    nn_time =  time.perf_counter() - initial_time
    nn_cost = calculates_distance(nn_route, distance_matrix)

    ga_runs = []
    aco_runs = []

    for seed in range(NUM_SEEDS):
        # GA
        random.seed(seed)
        initial_time = time.perf_counter()
        ga_route, ga_cost, ga_best_gen, ga_history, ga_iterations = genetic_algoritm(distance_matrix, time_limit=TIME_LIMIT)
        ga_runs.append({
            "seed": seed, 
            "cost": ga_cost, 
            "best_generation": ga_best_gen, 
            "history": ga_history, 
            "time": time.perf_counter() - initial_time,
            "iterations": ga_iterations, 
            "route": ga_route,
        })
        print(f"  semente {seed} | GA:  custo {ga_cost} em {ga_iterations} gerações")

        # ACO
        random.seed(seed)
        initial_time = time.perf_counter()
        aco_route, aco_cost, aco_best_gen, aco_history, aco_iterations = ant_colony_optimization(distance_matrix, time_limit=TIME_LIMIT)
        aco_runs.append({
            "seed": seed,
            "cost": aco_cost,
            "best_generation": aco_best_gen,
            "history": aco_history, 
            "time": time.perf_counter() - initial_time,
            "iterations": aco_iterations,
            "route": aco_route,
        })
        print(f"  semente {seed} | ACO: custo {aco_cost} em {aco_iterations} iterações")

    return {
        "label": label,
        "name": name,
        "num_cities": num_cities,
        "nn_cost": nn_cost,
        "nn_time": nn_time,
        "nn_gap": gap(nn_cost, opt_cost),
        "opt_cost": opt_cost,
        "ga": summarize(ga_runs, opt_cost),
        "aco": summarize(aco_runs, opt_cost),
    }

def print_table(results):
    """Prints the comparison table."""
    header = (f"{'Instância':<15}{'Cidades':>8}{'Ótimo':>9}{'NN':>9}{'NN gap':>8}"
              f"{'GA média':>10}{'GA gap':>8}{'ACO média':>11}{'ACO gap':>9}")
    print(header)
    print("-" * len(header))
 
    for r in results:
        print(f"{r['label']:<15}{r['num_cities']:>8}{r['opt_cost']:>9}{r['nn_cost']:>9}"
              f"{r['nn_gap']:>7.1f}%{r['ga']['mean_cost']:>10.0f}{r['ga']['mean_gap']:>7.1f}%"
              f"{r['aco']['mean_cost']:>11.0f}{r['aco']['mean_gap']:>8.1f}%")
 
    print(f"\nOrçamento de {TIME_LIMIT:.0f}s por execução, {NUM_SEEDS} sementes por técnica.")
    tempos_nn = ", ".join(f"{r['nn_time'] * 1000:.0f}ms" for r in results)
    print(f"O vizinho mais próximo roda em: {tempos_nn}.")
 
    print("\nIterações concluídas dentro do orçamento:")
    for r in results:
        print(f"  {r['label']:<15} GA: {r['ga']['mean_iterations']:>8.0f} gerações  |  "
              f"ACO: {r['aco']['mean_iterations']:>6.0f} iterações")
 
 
def save_table_csv(results, path="resultados.csv"):
    """Saves the consolidated table and the per seed detail."""
    with open(path, "w") as f:
        f.write("instancia,cidades,custo_otimo,custo_nn,gap_nn_pct,tempo_nn_s,"
                "tecnica,melhor,media,desvio_padrao,pior,gap_medio_pct,"
                "tempo_medio_s,iteracoes_medias,num_sementes\n")
        for r in results:
            for technique, data in (("GA", r["ga"]), ("ACO", r["aco"])):
                f.write(
                    f"{r['label']},{r['num_cities']},{r['opt_cost']},{r['nn_cost']},"
                    f"{r['nn_gap']:.2f},{r['nn_time']:.4f},"
                    f"{technique},{data['best_cost']},{data['mean_cost']:.1f},"
                    f"{data['stdev_cost']:.1f},{data['worst_cost']},{data['mean_gap']:.2f},"
                    f"{data['mean_time']:.2f},{data['mean_iterations']:.0f},{NUM_SEEDS}\n"
                )
 
    detail_path = path.replace(".csv", "_por_semente.csv")
    with open(detail_path, "w") as f:
        f.write("instancia,tecnica,semente,custo,iteracao_melhor,iteracoes,tempo_s\n")
        for r in results:
            for technique, data in (("GA", r["ga"]), ("ACO", r["aco"])):
                for run in data["runs"]:
                    f.write(f"{r['label']},{technique},{run['seed']},{run['cost']},"
                            f"{run['best_generation']},{run['iterations']},{run['time']:.2f}\n")
 
    return path, detail_path
 
 
def plot_convergence(results, path="convergencia.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ga_run = r["ga"]["median_run"]
        aco_run = r["aco"]["median_run"]

        for run, color, technique in ((ga_run, "Blue", "GA"), (aco_run, "green", "ACO")):
            total = len(run["history"])
            progress = [100.0 * i / total for i in range(total)]
            ax.plot(progress, run["history"], color=color, linewidth=2.0,
                    label=f"{technique} ({total} iterações)")
 
        ax.axhline(r["opt_cost"], color="red", linestyle="--", linewidth=1.2,
                   label=f"Ótimo conhecido ({r['opt_cost']})")
        ax.axhline(r["nn_cost"], color="grey", linestyle=":", linewidth=1.2,
                   label=f"Baseline NN ({r['nn_cost']})")
 
        ax.set_title(f"{r['label']} - {r['name']} ({r['num_cities']} cidades)")
        ax.set_xlabel(f"Progresso do orçamento de {TIME_LIMIT:.0f}s (%)")
        ax.set_ylabel("Custo da melhor rota")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
 
    fig.suptitle(
        f"Convergência com orçamento de {TIME_LIMIT:.0f}s por execução  |  "
        f"GA: {CROSSOVER_TYPE.upper()} + {MUTATION_TYPE}, pop={POPULATION_SIZE}  |  "
        f"ACO: {NUM_ANTS} formigas  |  mediana de {NUM_SEEDS} sementes"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
 
    return path

def main():
    results = []
    for label, tsp_path, opt_path in INSTANCES:
        print(f"Rodando instância {label} ({tsp_path}) com {NUM_SEEDS} sementes"
              f"e {TIME_LIMIT:.0f}s por execução...")
        results.append(run_instances(label, tsp_path, opt_path))
        
    print()
    print_table(results)

    table_path, detail_path = save_table_csv(results)
    figure_path = plot_convergence(results)

    print(f"\nTabelas salvas em: {table_path} e {detail_path}")
    print(f"Gráfico salvo em: {figure_path}")        

if __name__ == "__main__":
    main()