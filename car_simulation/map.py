import pygame
from .car import Car
import numpy as np

class Map():
    def __init__(self, x_limit, y_limit, car, background_color, track_color):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.car: Car = car
        self.environment = None
        self.background_color = background_color
        self.track_color = track_color
        self.surface = None
        self.checkpoints = []

    def load_environment(self, screen):
        if self.environment == None:
            screen.lock()
            colored_pixels = set()

            for x in range(screen.get_width()):
                for y in range(screen.get_height()):
                    color = screen.get_at((x,y))
                    if color != self.background_color:
                        colored_pixels.add((round(x),round(y)))
            screen.unlock()
            self.environment = colored_pixels

        return self.environment


    # allows user to draw the race track the car should learn to drive
    def draw_race_track(self, screen, car):
        self.surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        self.surface.fill((0,0,0,0))

        # objects for start button
        btn_rect = pygame.Rect(1100, 30, 150, 30)
        font = pygame.font.SysFont(None, 24)
        btn_text = font.render("Start Evolution", True, (255,255,255))

        running = True
        drawing = False

        counter = 0

        while running:

            screen.fill((255, 255, 255))
            screen.blit(self.surface, (0,0))

            # draw start button
            pygame.draw.rect(screen, "Blue", btn_rect)
            text_rect = btn_text.get_rect(center=btn_rect.center)
            screen.blit(btn_text, text_rect)

             # draw label
            if not drawing:
                font = pygame.font.SysFont("Arial", 16, bold=True)
                text_surface = font.render(f"Draw Racetrack with mouse. Make sure to draw in such a way that the car is on the track in the beginning.", True, (0, 0, 0))
                text_rect = text_surface.get_rect(topleft=(300, 300))
                screen.blit(text_surface, text_rect)


            # draw car at inital position
            self.draw_car(screen, car)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    running = False

            mouse_pos = pygame.mouse.get_pos()

            if btn_rect.collidepoint(mouse_pos):
                mouse_pressed = pygame.mouse.get_pressed()
                if mouse_pressed[0]:
                    return
            
            if drawing:
                pygame.draw.circle(self.surface, self.track_color, mouse_pos, 40)

                if counter % 20 == 0:
                    self.checkpoints.append(mouse_pos)
                counter += 1

            pygame.display.flip()
    
    # renders the map, updates the position of the car and shows information
    def render(self, screen: pygame.display, run_stat, cars):

        # overwrite anything
        screen.fill((255, 255, 255))

        if self.surface:
            screen.blit(self.surface, (0,0))

        # redraw the screen
        for car in cars:
            self.draw_car(screen, car["car"])

        self.draw_dashboard(screen, run_stat)


    def draw_car(self, screen: pygame.display, car):

        # draw car based car position and angle
        car_corners = car.get_car_corners()

        for i in range(0, 4):
            if i != 2:
                start_line = (car.x_position + car_corners[i % 4][0], car.y_position + car_corners[i % 4][1])
                end_line = (car.x_position + car_corners[(i+1) % 4][0], car.y_position + car_corners[(i+1) % 4][1])
            pygame.draw.line(screen, car.color, start_line, end_line, width=2)

        front_line_start = (car.x_position + car_corners[2][0], car.y_position + car_corners[2][1])
        front_line_end = (car.x_position + car_corners[3][0], car.y_position + car_corners[3][1])
        pygame.draw.line(screen, car.front_color, front_line_start, front_line_end, width=2)

    def draw_dashboard(self, screen, run_stat):

        x_position = 30
        y_position = 500

        font = pygame.font.SysFont("Arial", 16, bold=True)
        text_surface = font.render(f"Topology Evolution Data", True, (0, 0, 0))
        text_rect = text_surface.get_rect(topleft=(x_position, y_position))
        screen.blit(text_surface, text_rect)

        font = pygame.font.SysFont("Arial", 14)
        text_surface = font.render(f"Generation Amount: {run_stat.current_generation}", True, (0, 0, 0))
        text_rect = text_surface.get_rect(topleft=(x_position, y_position+20))
        screen.blit(text_surface, text_rect)  

        font = pygame.font.SysFont("Arial", 14)
        text_surface = font.render(f"Species Amount: {run_stat.current_species_amount}", True, (0, 0, 0))
        text_rect = text_surface.get_rect(topleft=(x_position, y_position+40))
        screen.blit(text_surface, text_rect)

        font = pygame.font.SysFont("Arial", 14)
        text_surface = font.render(f"Population: {run_stat.current_network_amount}", True, (0, 0, 0))
        text_rect = text_surface.get_rect(topleft=(x_position, y_position+60))
        screen.blit(text_surface, text_rect)

        if run_stat.best_network != None:
            font = pygame.font.SysFont("Arial", 14, bold=True)
            text_surface = font.render(f"Best Network:", True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(x_position, y_position+80))
            screen.blit(text_surface, text_rect)

            font = pygame.font.SysFont("Arial", 14)
            text_surface = font.render(f"Input Neurons: {len(run_stat.best_network.input_neurons)}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(x_position, y_position+100))
            screen.blit(text_surface, text_rect)

            font = pygame.font.SysFont("Arial", 14)
            text_surface = font.render(f"Hidden Neurons: {len(run_stat.best_network.hidden_neurons)}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(x_position, y_position+120))
            screen.blit(text_surface, text_rect)

            font = pygame.font.SysFont("Arial", 14)
            text_surface = font.render(f"Output Neurons: {len(run_stat.best_network.output_neurons)}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(x_position, y_position+140))
            screen.blit(text_surface, text_rect)

            font = pygame.font.SysFont("Arial", 14)
            text_surface = font.render(f"{run_stat.best_network}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(x_position, y_position+160))
            screen.blit(text_surface, text_rect)