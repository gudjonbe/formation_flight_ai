import pygame
from pygame.locals import *
import random
import os
import sys
import math
import neat
import numpy as np
import pickle

import visualize

pygame.init()

Draw = True

wash = np.load("project/data_draw.npy")




WIDTH, HEIGHT = 1000, 1000
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Formation Flight")


bg_img = pygame.image.load("project/wash_turb.png")
bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))


FPS = 30

WHITE = (255, 0, 255)
BLACK = (0, 0, 0)

SCORE_FONT = pygame.font.SysFont("freesansbold.ttf", 20)
FONT = pygame.font.SysFont("freesansbold.ttf", 50)


class Bird:
    VEL = 1
    VEL_x_init = 1#random.randint(-1,1)
    VEL_y_init = -1#random.randint(-1,1)
    COLOR = WHITE
    BIRD_ENERGY = 1000
    BIRD_RADIUS = 2

    def __init__(self, x=WIDTH // 2, y=HEIGHT // 2):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = self.BIRD_RADIUS
        self.x_vel = self.original_vel_x = self.VEL_x_init
        self.y_vel = self.original_vel_y = self.VEL_x_init
        self.bird_energy = self.BIRD_ENERGY

        self.LOST = False
        self.OUT_OF_ENERGY = False

        self.number_of_loops = 0
        self.color = (
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(150, 255),
        )

    def update(self):

        self.bird_is_lost()

        if self.bird_energy <= 0:
            self.OUT_OF_ENERGY = True

        else:
            self.bird_energy -= self.bird_wash()
            self.move()
            self.number_of_loops += 1

    def draw(self, win):
        
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)

    def move(self, up=False, down=False, right=False, left=False):
        if up and self.y_vel != -5:
            self.y_vel -= 1
        if down and self.y_vel != 5:
            self.y_vel += 1
        if right and self.x_vel != 5:
            self.x_vel += 1
        if left and self.x_vel != -5:
            self.x_vel -= 1

        if self.x + self.x_vel <= WIDTH and self.y + self.y_vel <= HEIGHT and self.x + self.x_vel >= 0 and self.y + self.y_vel >= 0:
            self.x +=  self.x_vel
            self.y +=  self.y_vel

    def bird_is_lost(self):
        lost = False
        #dist_lost = np.sqrt((WIDTH//2 - np.abs(self.x))**2 + (HEIGHT//2 - np.abs(self.y))**2)
        if self.x <= 10 or self.x >= WIDTH-10 or self.y <= 10 or self.y >= HEIGHT-10:
            self.LOST = True


    def bird_wash(self):
        if (self.x <= WIDTH) and (self.y <= HEIGHT):
            return 5*wash[self.x][self.y] + 0.1
        else:
            self.LOST = True
            return self.BIRD_ENERGY


def draw(win, bird, black=False):
    if black:
        win.fill(BLACK)
        WIN.blit(bg_img,(0,0))
    bird.draw(win)


def remove_bird(index):
    birds.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    global birds, ge, nets, points
    run = True
    #clock = pygame.time.Clock()
    points = 0

    birds = []
    ge = []
    nets = []

    def score():
        global points
        points += 1

    def statistics():
        global birds, ge
        text_1 = FONT.render(f"Birds Alive:  {str(len(birds))}", True, (123, 123, 123))
        text_2 = FONT.render(f"Generation:  {pop.generation + 1}", True, WHITE)

        WIN.blit(text_1, (50, 850))
        WIN.blit(text_2, (50, 900))

    for genome_id, genome in genomes:
        birds.append(Bird())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    delay = 1
    delay_ind = 0
    while run:

        
        delay_ind += 1
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for bird in birds:
            bird.update()
            if Draw:
                #pygame.draw.rect(WIN, BLACK, pygame.Rect(260, 850, 90, 90))
                draw(WIN, bird)

            score()

            
            if Draw:
                #statistics()
                
                pygame.display.update()

        if len(birds) == 0:
            if Draw:
                draw(WIN, bird, True)

            break
        
        
        for i, bird in enumerate(birds):
            
            if delay_ind%delay == 0:
                output = nets[i].activate((bird.x, bird.y, bird.x_vel, bird.y_vel))
                decision = output.index(max(output))

                if decision == 0:                
                    pass
                elif decision == 1:
                    bird.move(right=False, left=True)
                elif decision == 2:
                    bird.move(up=True, down=False)
                elif decision == 3:
                    bird.move(up=False, down=True)
                else:
                    bird.move(right=True, left=False)
                delay_ind = 0

                ge[i].fitness += 10**2

                if bird.LOST:
                    ge[i].fitness -= (10 - len(birds))**2
                    remove_bird(i)
                elif bird.OUT_OF_ENERGY:
                    ge[i].fitness -= (10 - len(birds))**2

                    remove_bird(i)
        

def run(config_path):
    global pop, delay, delay_ind

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)    
    #pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-18053')


    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(10000))
    

    

    winner = pop.run(eval_genomes, 10000)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
   # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=True, view=True)
    #visualize.plot_species(stats, view=True)

    

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
