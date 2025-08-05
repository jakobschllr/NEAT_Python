import math
import pygame

PI = math.pi

class Sensor():
    def __init__(self, position, sensor_reach):
        self.sensor_angle_offset = 0
        self.coordinate_offset = 0
        if position == "side-left":
            self.sensor_angle_offset = PI
            #self.coordinate_offset = -10
        elif position == "side-right":
            self.sensor_angle_offset = -2*PI
            #self.coordinate_offset = 10
        elif position == "front-left":
            self.coordinate_offset = -10
            self.sensor_angle_offset = -PI/2
        elif position == "front-right":
            self.coordinate_offset = 10
            self.sensor_angle_offset = -PI/2
        
        self.sensor_reach = sensor_reach
    

    def get_distance(self, car_angle, car_coordinates, environment, screen):
        sensor_angle = car_angle + self.sensor_angle_offset

        obstacle_distance = 0
        
        sensor_coordinates = (car_coordinates[0] - (math.cos(car_angle) * self.coordinate_offset), car_coordinates[1] - (math.sin(car_angle) * self.coordinate_offset))

        # draw sensor line for animation
        def draw_sensor_line():
            start_line = sensor_coordinates
            end_line = (sensor_coordinates[0]-(math.cos(sensor_angle)*self.sensor_reach), sensor_coordinates[1]-(math.sin(sensor_angle) * self.sensor_reach))
            pygame.draw.line(screen, "Red", start_line, end_line, width=1)
            pygame.display.flip()
            found_obstacle = False

        #draw_sensor_line()

        cos = math.cos(sensor_angle)
        sin = math.sin(sensor_angle)

        for measurement_point in range(self.sensor_reach):
            x = cos * measurement_point
            y = sin * measurement_point
            
            found_obstacle = self.check_for_obstacles((sensor_coordinates[0]-x, sensor_coordinates[1]-y), environment)
            if found_obstacle:
                if obstacle_distance < 5: # car (almost) hit obstacle
                    return obstacle_distance, True
                return obstacle_distance, False
            else:
                obstacle_distance += 1

        if not found_obstacle:
            return self.sensor_reach, False
    
    def check_for_obstacles(self, coordinates, environment):
        coordinates = (round(coordinates[0]), round(coordinates[1]))
        if coordinates not in environment:
            return True
        return False

