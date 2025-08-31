from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker
from .species import Species
import json
import copy

with open("config_car.json", "r") as f:
    config = json.load(f)

c1 = config["c1"] # parameter to adjust influence of adjoint gene amount in compatibilty distance calculation
c2 = config["c2"] # parameter to adjust influence of excess gene amount in compatibilty distance calculation
c3 = config["c3"] # parameter to adjust influence of average weight difference in compatibilty distance calculation
species_survival_rate = config["species_survival_rate"] # percentage of fittest networks in each species that survive each generation
elit_nets_amount = config["elit_nets_amount"] # amount of fittest networks, that are elit and are copied to the next generation
mutation_offspring_rate = config["mutation_offspring_rate"] # amount of species children that are created by mutations
crossover_offspring_rate = config["crossover_offspring_rate"] # amount of species children that are created by crossovers
weight_perturbation_prob = config["weight_perturbation_prob"] # chance of a connection weight being mutated with perturbation
weight_mutation_random_value_prob = config["weight_mutation_random_value_prob"] # chance of a connection weight being mutated by setting it to a random value
bias_weight_mutation_prob = config["bias_weight_mutation_prob"] # chance of a bias in a random neuron in a network being perturbarted

gene_disabled_rate = config["gene_disabled_rate"] # chance of inherited connection staying disabled if it is disabled in either parent (crossover)
new_neuron_prob = config["new_neuron_prob"] # chance a new hidden neuron is added to a network
new_connection_prob = config["new_connection_prob"] # chance of a new connectino being added to a network
species_similarity_threshold = config["species_similarity_threshold"] # threshold for compatibilty between networks (the higher the treshold the more networks are in a species)
stagnation_treshold = config["stagnation_treshold"] # amount of generations a species survives without fitness improvements

class PopulationHandler():
    def __init__(self, initial_net_amount, input_neurons, output_neurons, max_generations, fitness_function, run_stat, fitness_function_multiple_nets, target_fitness=1.0):
        self.initial_net_amount = initial_net_amount
        self.network_amount = initial_net_amount
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.max_generations = max_generations
        self.species: list[Species] = []
        self.hist_marker = HistoricalMarker(self.input_neurons + self.output_neurons) # hidden neuron IDs start after IDs for input and output neurons
        self.fitness_function = fitness_function
        self.fitness_function_multiple_nets = fitness_function_multiple_nets
        self.run_stat = run_stat
        self.current_population = []
        self.generation_counter = 1
        self.target_fitness = target_fitness

    def calculate_raw_fitness(self):
        # fitness function that takes whole population
        if self.fitness_function_multiple_nets:
            self.fitness_function(self.current_population)
            
        # fitness function that takes each network separately
        else:
            for net in self.current_population:
                self.fitness_function(net)

    def initial_population(self):
        # create first species
        initial_species = Species(c1, c2, c3, species_survival_rate)

        # create networks and add to first species
        for _ in range(0, self.network_amount):
            network = Network(self.input_neurons, self.output_neurons)
            network.initialize_minimal_network(self.hist_marker)
            initial_species.add_network(network)
            self.current_population.append(network)

        self.calculate_raw_fitness() # runs simulation, calculates raw fitness and saves it to net.raw_fitness

        initial_species.calculate_fitness(self.run_stat)
        # calculate adjusted fitness for each network in first species and remove low performing
        self.species.append(initial_species)
        print(f"Initial Species with {self.network_amount} organisms populated")


    def start_evolution_process(self):

        best_network = None
        while self.generation_counter < self.max_generations and (best_network == None or best_network.raw_fitness < self.target_fitness):

            # get sum of all average adjusted fitnesses of all species
            adjusted_fitness_all_species = 0
            for species in self.species[:]:

                species_removed = False

                if len(species.networks) == 0:
                    self.species.remove(species)
                    species_removed = True
                else:
                    species.calculate_fitness(self.run_stat)
                    species.update_representative()

                #adjust stagnation counter for species
                species.adjust_stagnation_counter()

                #if species fitness stagnated to long remove species
                if species.stagnation_counter == stagnation_treshold and not species_removed:
                    self.species.remove(species)

                else:
                #remove low performing networks from species
                    species.remove_lowperforming_networks()
                    adjusted_fitness_all_species += species.total_adjusted_fitness     

            new_population = []

            # each species produces certain amount of children based on its adjusted fitness

            for species in self.species:
                species_children_amount = round((species.total_adjusted_fitness / adjusted_fitness_all_species) * self.network_amount)
                children, elit_networks = species.produce_offspring(
                    species_children_amount,
                    mutation_offspring_rate,
                    weight_perturbation_prob,
                    weight_mutation_random_value_prob,
                    bias_weight_mutation_prob,
                    crossover_offspring_rate,
                    new_neuron_prob,
                    new_connection_prob,
                    self.hist_marker,
                    elit_nets_amount,
                    gene_disabled_rate
                )

                if children != None and elit_networks != None:
                    new_population = new_population + children
                    for net in elit_networks:
                        if best_network == None or net.raw_fitness > best_network.raw_fitness:
                            best_network = copy.deepcopy(net) 

            # update population calculate raw_fitness for each new network
            self.current_population = new_population
            self.calculate_raw_fitness()

            # reset species
            for species in self.species:
                species.reset_species()

            # all newly produced children have to be assigned to a species based on their compatibilty distance
            for network in new_population:
                found_species = False

                for species in self.species:
                    compatibility_dist = species.calculate_compatibility_distance(network)
                    if compatibility_dist < species_similarity_threshold:
                        species.add_network(network)
                        found_species = True
                        break
                if not found_species:
                    # create new species
                    new_species = Species(c1, c2, c3, species_survival_rate)
                    new_species.add_network(network)
                    self.species.append(new_species)                

            self.run_stat.update(self.generation_counter, len(self.species), len(new_population), best_network)
            self.run_stat.print()
 
            self.generation_counter += 1


        print("Best network after ", self.max_generations, " generations.")
        return best_network
