from neat_classes.PopulationHandler import PopulationHandler
from car_simulation.test import test_simulation
from car_simulation.SimulationHandler import SimulationHandler
from neat_classes.RuntimeStatus import RuntimeStatus
import sys

# Object to track runtime status
run_stat = RuntimeStatus()


# entry point for program
def main():

    if len(sys.argv) > 1:
        param = sys.argv[1]

        if param == "xor":

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
                max_generations=100,
                fitness_function=fitness_function_xor,
                run_stat=run_stat,
                fitness_function_multiple_nets=False, # if True, then fitness function calculcates fitness for all networks, if False then only for one network
            )
            population_handler.initial_population()
            best_net = population_handler.start_evolution_process()

            print("Best network: ", best_net)
            print("Input   |   Raw Output   |   Rounded Ouput")
            print(f" 0 0    |       {round(best_net.compute_inputs(0,0)[0], 2)}     |   {round(best_net.compute_inputs(0,0)[0])}")
            print(f" 0 1    |       {round(best_net.compute_inputs(0,1)[0], 2)}     |   {round(best_net.compute_inputs(0,1)[0])}")
            print(f" 1 0    |       {round(best_net.compute_inputs(1,0)[0], 2)}     |   {round(best_net.compute_inputs(1,0)[0])}")
            print(f" 1 1    |       {round(best_net.compute_inputs(1,1)[0], 2)}     |   {round(best_net.compute_inputs(1,1)[0])}")

        
        elif param == "car":

            # fitness function 2D car
            simulation_handler = SimulationHandler(run_stat)
            def fitness_function_car(population): # input parameter can be a single network or a population of networks
                simulation_handler.start_episode(population)
            
            population_handler = PopulationHandler(
                initial_net_amount=150,
                input_neurons=4,
                output_neurons=2,
                max_generations=100,
                fitness_function=fitness_function_car,
                run_stat=run_stat,
                fitness_function_multiple_nets=True, # if True, then fitness function calculcates fitness for all networks, if False then only for one network
            )
            simulation_handler.initialize()
            population_handler.initial_population()
            population_handler.start_evolution_process()

        else:
            print("Invalid problem definition. Write xor or car.")
            return
        
    else:
        print("Invalid amount of parameters.")

main()