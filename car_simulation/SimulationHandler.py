import pygame
from .map import Map
from .car import Car
import time
import math

PI = math.pi

class SimulationHandler():
    def __init__(self, run_stat):
        self.width_px = 1280
        self.height_px = 720
        self.car_start_position = (600,600)
        self.car = Car(self.car_start_position)
        self.map = Map(self.width_px, self.height_px, self.car, (255, 255, 255), (128, 128, 128))
        pygame.init()
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()
        self.run_stat = run_stat

    def initialize(self):
        self.map.draw_race_track(self.screen, Car(self.car_start_position))

    # this method is called from the get_raw_fitness method in the network class from neat, it returns the fitness of the network
    def start_episode(self, networks):

        start_time = time.time()

        cars = []
        for net in networks:
            cars.append({
                "car": Car(self.car_start_position),
                "net": net,
                "distance_traveled": 0,
                "last_pos": None,
                "cur_pos": None,
                "stagnation_counter": 0,
                "is_active": True,
                "checkpoint_counter": 0
                })
    
        # get map environment data
        environment = self.map.load_environment(self.screen)

        running = True

        frames = 0

        while running:
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            running = False
            current_time = time.time()

            # load current sensor data from cars
            for car in cars:
                if car["is_active"]:
                    running = True
                    sensor_data = car["car"].get_sensor_data(environment, self.screen)

                    # feed sensor data to network and generate output
                    in_1 = sensor_data[0] / self.car.sensor_reach
                    in_2 = sensor_data[1] / self.car.sensor_reach
                    in_3 = sensor_data[2] / self.car.sensor_reach
                    in_4 = sensor_data[3] / self.car.sensor_reach
                    output = car["net"].compute_inputs(in_1, in_2, in_3, in_4)
                    speed = output[0]
                    steering = output[1]

                    # move car based on networks decision
                    car["car"].steer(2*steering-1) # scale value to be between -1 and 1
                    car["car"].move_forward(speed)

                    car["last_pos"] = car["cur_pos"]
                    car["cur_pos"] = (round(car["car"].x_position), round(car["car"].y_position))

                    distance = car["car"].get_euclidean_distance(car["last_pos"], car["cur_pos"])

                    # update if car came close to checkpoint
                    old_checkpoint_counter = car["checkpoint_counter"]
                    if car["car"].hit_checkpoint(self.map.checkpoints):
                        car["checkpoint_counter"] += 1

                    # check if car movement stagnates
                    if old_checkpoint_counter != car["checkpoint_counter"]:
                        car["stagnation_counter"] = 0
                    else:
                        car["stagnation_counter"] += 1

                    # survival_time = current_time - start_time
            
                    # handle network fitness tracking
                    if car["car"].hit_obstacle:
                        car["is_active"] = False

                        checkpoints_reached = car["checkpoint_counter"] / len(self.map.checkpoints)
                        steering_fluidity = (1 - car["car"].steering_tracker["avg"] / PI )
                        obstacle_distance = car["car"].obstacle_distance_tracker["avg"] / car["car"].sensor_reach
                        # car["distance_traveled"] distance traveled euclidean
                        # survival_time

                        car["net"].raw_fitness =  (checkpoints_reached + steering_fluidity + 0.7 * obstacle_distance) / 3
                        
                    elif car["stagnation_counter"] == 20:
                        car["distance_traveled"] = 0
                        car["checkpoint_counter"] = 0
                        car["is_active"] = False
                        car["net"].raw_fitness = 0
                    else:
                        car["distance_traveled"] += distance

            # rerender map
            self.map.render(self.screen, self.run_stat, cars)
        
            # flip() the display to put your work on screen
            pygame.display.flip()

            self.clock.tick(60)  # limits FPS to 60
            frames += 1
