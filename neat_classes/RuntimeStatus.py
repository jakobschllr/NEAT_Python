
class RuntimeStatus():    
    def __init__(self):
        self.current_generation = 0
        self.current_species_amount = 1
        self.current_network_amount = 0
        self.best_network = None
        self.data = []

    def update(self, current_generation, current_species_amount, current_network_amount, best_network):
        self.current_generation = current_generation
        self.current_species_amount = current_species_amount
        self.current_network_amount = current_network_amount
        self.best_network = best_network
        self.data.append((self.current_generation, self.best_network.raw_fitness))

    def print(self):
        print(f"********** Generation {self.current_generation} **********")
        print(f"Population has {self.current_species_amount} species.")
        print(f"Population has {self.current_network_amount} networks.")
        print(f"Best network: {self.best_network}")

    def reset(self):
        self.__init__()