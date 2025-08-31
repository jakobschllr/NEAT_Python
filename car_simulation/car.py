import math
import numpy as np
from .sensor import Sensor
import pygame

PI = math.pi

class Car():
    def __init__(self, starting_point):
        self.angle = PI + (PI/2) # im Bogenmaß
        self.x_position = starting_point[0]
        self.y_position = starting_point[1]
        self.color = "Blue"
        self.front_color = "Red"
        self.width = 15
        self.height = 30
        self.sensor_reach = 45
        self.sensors = {
            "s1": Sensor("front-left", self.sensor_reach),
            "s2": Sensor("front-right", self.sensor_reach),
            "s3": Sensor("side-left", self.sensor_reach),
            "s4": Sensor("side-right", self.sensor_reach)
        }
        self.hit_obstacle = False
        self.passed_checkpoints = set()

        self.steering_tracker = {
            "amt": 0,
            "sum": 0,
            "avg": 0,
            "counter": 0,
            "last_angle": 0
        }

        self.obstacle_distance_tracker = {
            "amt": 0,
            "sum": 0,
            "avg": 0,
        }



    def get_car_corners(self):
        corner_1 = np.array([ -self.width, -self.height])
        corner_2 = np.array([self.width, -self.height])
        corner_3 = np.array([self.width, self.height])
        corner_4 = np.array([-self.width, self.height])

        rotation_matrix = np.array([[math.cos(self.angle), -math.sin(self.angle)],
                  [math.sin(self.angle), math.cos(self.angle)]
                  ])

        rotated_corner_1 = np.dot(rotation_matrix, corner_1)
        rotated_corner_2 = np.dot(rotation_matrix, corner_2)
        rotated_corner_3 = np.dot(rotation_matrix, corner_3)
        rotated_corner_4 = np.dot(rotation_matrix, corner_4)

        return [rotated_corner_1, rotated_corner_2, rotated_corner_3, rotated_corner_4]


    def move_forward(self, speed):
        angle = self.angle - PI/2

        self.x_position -= math.cos(angle) * (speed*20)
        self.y_position -= math.sin(angle) * (speed*20)

    def steer(self, value):
        # value goes from -1.0 (left) to 1.0 (right)
        self.angle += value * math.pi / 16
        self.update_steering_tracker(self.angle)

    def update_steering_tracker(self, current_angle):
        self.steering_tracker["counter"] += 1
        if self.steering_tracker["counter"] == 15:
            self.steering_tracker["sum"] += abs(self.steering_tracker["last_angle"] - current_angle)
            self.steering_tracker["amt"] += 1
            self.steering_tracker["avg"] = self.steering_tracker["sum"] / self.steering_tracker["amt"]
            self.steering_tracker["counter"] = 0
            self.steering_tracker["last_angle"] = current_angle

    def get_sensor_data(self, environment, screen):
        sensor_data = []
        for sensor in self.sensors.values():
            distance, hit_obstacle = sensor.get_distance(self.angle, (self.x_position, self.y_position), environment, screen)
            sensor_data.append(distance)
            if hit_obstacle:
                self.hit_obstacle = True
        self.update_obstacle_distance_tracker(sensor_data)
        return sensor_data
    
    def update_obstacle_distance_tracker(self, distances):
            avg = sum(dist for dist in distances) / 4
            self.obstacle_distance_tracker["amt"] += 1
            self.obstacle_distance_tracker["sum"] += avg
            self.obstacle_distance_tracker["avg"] = self.obstacle_distance_tracker["sum"] / self.obstacle_distance_tracker["amt"]
    
    def get_euclidean_distance(self, pos_old, pos_new):
        if pos_old != None and pos_new != None:
            x_distance = pos_new[0] - pos_old[0]
            y_distance = pos_new[1] - pos_old[1]
            return math.sqrt(x_distance**2 + y_distance**2)
        return 1
    
    def hit_checkpoint(self, checkpoints):
        for (x,y) in checkpoints:
            distance = self.get_euclidean_distance((x,y), (self.x_position, self.y_position))
            if distance <= self.width and (x,y) not in self.passed_checkpoints:
                self.passed_checkpoints.add((x,y))
                return True
        return False
