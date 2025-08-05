from .SimulationHandler import SimulationHandler
import math

def test_simulation():
    simulation_handler = SimulationHandler()
    simulation_handler.initialize()
    simulation_handler.start_episode()
