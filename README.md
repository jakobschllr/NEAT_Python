# NEAT (Neuro Evolution of Augmenting Topologies)

## Overview
This project is a python implementation of the NEAT method (Neuro Evolution of Augmenting Topologies) by Stanley and Miikkulaien. NEAT can be used to find neural networks within an exploration space in order to solve predefined optimization problems. Besides default python libraries and numpy the projects does not use other ml-related dependencies.

## Quick start
To try the algorithm you can choose one of two predefined problems to solve. The first one is XOR and the second one is a 2D-Car that learns to drive a racetrack in a pygame environment. After downloading the repository execute `python3 init_neat.py xor` or `python3 init_neat.py car` to start the evolution process. To adjust the evolution process you can change the parameters in the config.json file (more details about this file later).

## Use NEAT for your own problems
To use this NEAT implementation on your own set of problems you have to create an instance of the `PopulationHandler` class which is localed in the neat_classes directory. This class requires initial parameters that need to be defined:

```
population_handler = PopulationHandler(
                initial_net_amount=150,
                input_neurons=4,
                output_neurons=2,
                max_generations=100,
                run_stat=run_stat,
                fitness_function=fitness_function_xor,
                fitness_function_multiple_nets=True,
```

`initial_net_amount` is the amount of networks in the ininital population. During the evolution the size of the population will not change dramatically from the size of the initial population.
`input_neurons` and `output_neurons` is the amount of input and output neurons for all networks. This numbers depend on the problem you want to solve.
`max_generations` defines the amount of total generations for the networks to evolve.
`run_stat` expects an instance of the RuntimeStatus class from the neat_classes directory. This object is responsible for tracking data during the evolution.
`fitness_function` is a function that you need to define based on the problem you want to solve. This is where you calculate how good a network performed when trying to solve your problem. Based on the problem you want to solve, you can have two different kinds of implementations for the fitness function. If your fitness function takes the whole current population and calculates the fitness of all networks at once (I used that for the 2D car animation, where I want all cars to move at the same time), your fitness function receives a list containing all networks of the current population. A network is an instance of the Network class from network.py. If you want to use this implementation, you need to set the parameter `fitness_function_multiple_nets` as True. The other case would be that your fitness function only calculates the fitness for one network at the time. In that case set `fitness_function_multiple_nets` to False. In both cases your fitness function needs to save the fitness for each network in the `raw_fitness` attribute of the network object.

Here is an example for a fitness function that calculates the fitness for one network:
```
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
```

After defining a fitness function and creating the PopulationHandler object, you can start the evolution process with
```
population_handler.initial_population()
best_net = population_handler.start_evolution_process()
```

After completing the defined amount of generations, the `start_evolution_process`-method of the population_handler returns the best network that was discovered so far. To see the outputs this network generates, you can use the `compute_inputs`-method of the network object. Here's an example:
```
            print("Best network: ", best_net)
            print("Input   |   Raw Output   |   Rounded Ouput")
            print(f" 0 0    |       {round(best_net.compute_inputs(0,0)[0], 2)}     |   {round(best_net.compute_inputs(0,0)[0])}")
            print(f" 0 1    |       {round(best_net.compute_inputs(0,1)[0], 2)}     |   {round(best_net.compute_inputs(0,1)[0])}")
            print(f" 1 0    |       {round(best_net.compute_inputs(1,0)[0], 2)}     |   {round(best_net.compute_inputs(1,0)[0])}")
            print(f" 1 1    |       {round(best_net.compute_inputs(1,1)[0], 2)}     |   {round(best_net.compute_inputs(1,1)[0])}")
```

`compute_inputs` returns a list that contains the output of each output neuron.


## Config.json
To tweak the evolution process you can adjust the parameter settings in the config.json file.
```
{
    "c1": 1.0,
    "c2": 1.0,
    "c3": 0.4,
    "species_similarity_threshold": 3.0,

    "species_survival_rate": 20,
    "elit_nets_amount": 2,
    "stagnation_treshold": 20,

    "mutation_offspring_rate": 60,
    "crossover_offspring_rate": 40,

    "weight_perturbation_prob": 10,
    "weight_mutation_random_value_prob": 10,
    "bias_weight_mutation_prob": 10,
    
    "gene_disabled_rate": 75,
    "new_neuron_prob": 20,
    "new_connection_prob": 30
}
```
