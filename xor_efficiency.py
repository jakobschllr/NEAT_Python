from neat_classes.PopulationHandler import PopulationHandler
from car_simulation.test import test_simulation
from car_simulation.SimulationHandler import SimulationHandler
from neat_classes.RuntimeStatus import RuntimeStatus
import sys
import matplotlib.pyplot as plt
import numpy as np

# Object to track runtime status
run_stat = RuntimeStatus()


# entry point for program
def main():
        # fitness function xor
        def fitness_function_xor(net):
            outputs = [
                    (net.compute_inputs(0,0)[0], 0), 
                    (net.compute_inputs(0,1)[0], 1),
                    (net.compute_inputs(1,0)[0], 1),
                    (net.compute_inputs(1,1)[0], 0)
                ]

            total_error = 0
            for out, target in outputs:
                total_error += (out-target)**2
            net.raw_fitness = 1 / (1+total_error)


        population_handler = PopulationHandler(
            initial_net_amount=150,
            input_neurons=2,
            output_neurons=1,
            max_generations=300,
            fitness_function=fitness_function_xor,
            run_stat=run_stat,
            fitness_function_multiple_nets=False, # if True, then fitness function calculcates fitness for all networks, if False then only for one network
            target_fitness=0.9
        )
        population_handler.initial_population()
        best_net = population_handler.start_evolution_process()


        def print_best_network():
            print("Best network: ", best_net)
            print("Input   |   Raw Output   |   Rounded Ouput")
            print(f" 0 0    |       {round(best_net.compute_inputs(0,0)[0], 2)}     |   {round(best_net.compute_inputs(0,0)[0])}")
            print(f" 0 1    |       {round(best_net.compute_inputs(0,1)[0], 2)}     |   {round(best_net.compute_inputs(0,1)[0])}")
            print(f" 1 0    |       {round(best_net.compute_inputs(1,0)[0], 2)}     |   {round(best_net.compute_inputs(1,0)[0])}")
            print(f" 1 1    |       {round(best_net.compute_inputs(1,1)[0], 2)}     |   {round(best_net.compute_inputs(1,1)[0])}")

        print_best_network()

def calculate_avg_fitness_after_n_generations():
    """
    Calculates the average fitness of the best networks in each generation
    """
    sum_vector = None
    sample_size = 20

    for _ in range(sample_size):
        main()
        data = np.array(run_stat.data)
        sum_vector = sum_vector + data if sum_vector is not None else data
        run_stat.reset()

    avg_vector = sum_vector * (1/sample_size)
    print(avg_vector)

    x_values = [p[0] for p in avg_vector]
    y_values = [p[1] for p in avg_vector]

    plt.plot(x_values, y_values, marker="o", label="NEAT Average Fitness over 100 Generations")

    # Achsenbereiche festlegen
    plt.xlim(0, 100)
    plt.ylim(0, 1.0)

    # Achsenbeschriftung und Legende
    plt.xlabel("Generations")
    plt.ylabel("Average Network Fitness")
    plt.legend()

    plt.show()


def calc_generation_amount_for_high_fitness():
    """
    Calculates the amount of generations it takes for 100 networks to reach a fitness > 0.9
    """
    sample_size = 100
    data = []

    for i in range(sample_size):
        main()
        data.append(run_stat.current_generation)
        run_stat.reset()

    
    plt.bar([i for i in range(1, sample_size+1)], data)
    plt.xlabel("Trainings")
    plt.ylabel("Generations")
    plt.title("Amount of generations to find net with fitness > 90%")

    plt.show()

    print("Avergage generation amount to reach fitness level: ", sum(data) / sample_size)

calc_generation_amount_for_high_fitness()