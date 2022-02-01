#################
##### PONG  #####
#################
# pip install pygame

import pygame
import numpy as np
from time import sleep

from config import *


class Pong:
    def __init__(self, FRAME_RATE,
                 SCREEN_BG_PATH, SCREEN_START, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_GROUND,
                 FONT_PATH, FONT_SIZE, FONT_COLOR,
                 SCORE1_POS, SCORE2_POS,
                 P1_IMAGE_PATH, P1_WIDTH, P1_HEIGHT, P1_STEP_SIZE, P1_X, p1_y,
                 P2_IMAGE_PATH, P2_WIDTH, P2_HEIGHT, P2_STEP_SIZE, P2_X, p2_y,
                 B_COLOR, B_RADIUS, b_x_step_size, B_X_MAX_STEP_SIZE, b_y_step_size, B_Y_MAX_STEP_SIZE, b_x, b_y):
        # Init
        pygame.init()

        # Frame rate
        self.FRAME_RATE = FRAME_RATE

        # Screen
        self.SCREEN_BG = pygame.image.load(SCREEN_BG_PATH)
        self.SCREEN_START = SCREEN_START
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_GROUND = SCREEN_GROUND
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Font
        # main font for scoreboard
        self.FONT = pygame.font.Font(FONT_PATH, FONT_SIZE)
        # main font with smaller size for training statistics
        self.S_FONT = pygame.font.Font(FONT_PATH, FONT_SIZE // 2)
        self.FONT_COLOR = FONT_COLOR

        # Scoreboard
        self.SCORE1_POS = SCORE1_POS
        self.SCORE2_POS = SCORE2_POS

        # Match statistics
        self.match_time = 0
        self.last_match_time = 0  # time played in last match
        self.match_count = 0      # number of matches played
        self.match_touches = 0    # number of times the AI hits the ball
        self.last_match_touches = 0

        # Game statistics
        self.game_p1_score = 0
        self.game_p2_score = 0

        # Player 1: AI
        self.P1_IMAGE = pygame.image.load(P1_IMAGE_PATH)
        self.P1_WIDTH = P1_WIDTH
        self.P1_HEIGHT = P1_HEIGHT
        self.P1_STEP_SIZE = P1_STEP_SIZE
        self.P1_X = P1_X
        self.p1_y = p1_y
        self.P1_STARTX = P1_X
        self.P1_STARTY = p1_y
        # Initial offset from p1_y (as long as key is pressed, we move)
        self.p1_offset = 0
        self.p1_active_movements = {
            "up": False,
            "down": False
        }

        # Player 2: YOU | AI
        self.P2_IMAGE = pygame.image.load(P2_IMAGE_PATH)
        self.P2_WIDTH = P2_WIDTH
        self.P2_HEIGHT = P2_HEIGHT
        self.P2_STEP_SIZE = P2_STEP_SIZE
        self.P2_X = P2_X
        self.p2_y = p2_y
        self.P2_STARTX = P2_X
        self.P2_STARTY = p2_y
        # Initial offset from p2_y (as long as key is pressed, we move)
        self.p2_offset = 0
        self.p2_active_movements = {
            "up": False,
            "down": False
        }

        # Ball
        self.B_COLOR = B_COLOR
        self.B_RADIUS = B_RADIUS
        self.INITIAL_B_X_STEP_SIZE = b_x_step_size
        self.INITIAL_B_Y_STEP_SIZE = b_y_step_size
        self.b_x_step_size = b_x_step_size
        self.B_X_MAX_STEP_SIZE = B_X_MAX_STEP_SIZE
        self.b_y_step_size = b_y_step_size
        self.B_Y_MAX_STEP_SIZE = B_Y_MAX_STEP_SIZE
        self.b_x = b_x
        self.b_y = b_y
        self.B_STARTX = b_x
        self.B_STARTY = b_y

    # Play game (set of matches)
    def run(self, n_play_matches=0, n_generations=0, p_size=0, n_train_matches=0):

        # Set Clock
        clock = pygame.time.Clock()

        # On train mode: x2 if we evaluate fitness for both parents and children on every generation (thus we have double the amount of games)
        matches_per_generation = p_size * n_train_matches * 2

        # While we have matches left to play
        while self.match_count < n_play_matches or self.match_count < n_generations * matches_per_generation:
            for event in pygame.event.get():

                # If QUIT, quit
                if event.type == pygame.QUIT:
                    self.match_count = max(
                        n_play_matches, n_generations * matches_per_generation)

                # If Play mode: You play Vs AI. You control Player 2. AI is controlled from outside by NN
                if self.match_count < n_play_matches:
                    # Update Player 2 offset on Key Down
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.p2_active_movements["up"] = True
                            self.p2_offset = -self.P2_STEP_SIZE
                        if event.key == pygame.K_DOWN:
                            self.p2_active_movements["down"] = True
                            self.p2_offset = self.P2_STEP_SIZE
                    # Update Player 2 offset on Key Up
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_UP:
                            self.p2_active_movements["up"] = False
                            if self.p2_active_movements["down"] == False:
                                self.p2_offset = 0
                            else:
                                self.p2_offset = self.P2_STEP_SIZE
                        if event.key == pygame.K_DOWN:
                            self.p2_active_movements["down"] = False
                            if self.p2_active_movements["up"] == False:
                                self.p2_offset = 0
                            else:
                                self.p2_offset = -self.P2_STEP_SIZE

                # If Train mode: AI plays Vs AI. AI is controlled from outside by NN
                else:
                    pass

            # Update Player 1 position
            self.p1_y += self.p1_offset
            if self.p1_y > self.SCREEN_GROUND - self.P1_HEIGHT:
                # keep within bounds (Player & Environment Collision)
                self.p1_y = self.SCREEN_GROUND - self.P1_HEIGHT
            if self.p1_y < self.SCREEN_START:
                # keep within bounds (Player & Environment Collision)
                self.p1_y = self.SCREEN_START

            # Update Player 2 position
            self.p2_y += self.p2_offset
            if self.p2_y > self.SCREEN_GROUND - self.P2_HEIGHT:
                # keep within bounds (Player & Environment Collision)
                self.p2_y = self.SCREEN_GROUND - self.P2_HEIGHT
            if self.p2_y < self.SCREEN_START:
                # keep within bounds (Player & Environment Collision)
                self.p2_y = self.SCREEN_START

            # Update Ball position
            self.b_x += self.b_x_step_size
            self.b_y += self.b_y_step_size
            if self.b_y >= self.SCREEN_GROUND - self.B_RADIUS:
                # keep within bounds (Ball & Environment Collision)
                self.b_y_step_size = -self.b_y_step_size
            if self.b_y <= self.SCREEN_START + self.B_RADIUS:
                # keep within bounds (Ball & Environment Collision)
                self.b_y_step_size = -self.b_y_step_size

            # Update Score & Match count. Reset match
            if self.b_x <= self.SCREEN_START:
                self.game_p2_score += 1
                self.reset()
                self.match_count += 1
            if self.b_x >= self.SCREEN_WIDTH:
                self.game_p1_score += 1
                self.reset()
                self.match_count += 1

            # Graphics
            # Game BG
            self.screen.blit(self.SCREEN_BG,
                             [self.SCREEN_START, self.SCREEN_START])

            # Player 1 Paddle
            p1 = self.screen.blit(self.P1_IMAGE, [self.P1_X, self.p1_y])

            # Player 2 Paddle
            p2 = self.screen.blit(self.P2_IMAGE, [self.P2_X, self.p2_y])

            # Ball
            b = pygame.draw.circle(self.screen, self.B_COLOR,
                                   (self.b_x, self.b_y), self.B_RADIUS)

            # Player 1: AI
            self.screen.blit(self.FONT.render(
                "AI", False, self.FONT_COLOR), self.SCORE1_POS)

            # Player 1 Score: AI Score
            self.screen.blit(self.FONT.render(
                str(self.game_p1_score), False, self.FONT_COLOR), (self.SCORE1_POS[0], self.SCORE1_POS[1] + 36))

            # If Play mode
            if self.match_count < n_play_matches:

                # Player 2: YOU
                self.screen.blit(self.FONT.render(
                    "YOU", False, self.FONT_COLOR), self.SCORE2_POS)

                # Player 2 Score: YOU(R) Score
                self.screen.blit(self.FONT.render(
                    str(self.game_p2_score), False, self.FONT_COLOR), (self.SCORE2_POS[0] + 10, self.SCORE2_POS[1] + 36))

            # If Train mode
            if self.match_count < n_generations * matches_per_generation:

                # Player 2: AI
                self.screen.blit(self.FONT.render(
                    "AI", False, self.FONT_COLOR), self.SCORE2_POS)

                # Player 2 Score: AI Score
                self.screen.blit(self.FONT.render(
                    str(self.game_p2_score), False, self.FONT_COLOR), (self.SCORE2_POS[0], self.SCORE2_POS[1] + 36))

                # Generation
                gen = self.match_count // matches_per_generation + 1
                self.screen.blit(self.S_FONT.render(
                    "Gen {}".format(gen), False, self.FONT_COLOR),
                    (self.SCREEN_WIDTH // 2 - 30, self.SCORE1_POS[1] + 45))

                # Touches (more is usually better. AI is learning)
                self.screen.blit(self.S_FONT.render(
                    "Score {}".format(self.match_time), False, self.FONT_COLOR),
                    (self.SCREEN_WIDTH // 2 - 50, self.SCORE1_POS[1] + 25))

            # Update the contents of the entire display
            pygame.display.flip()

            # Pygame - handled collisions
            if b.colliderect(p1):
                if self.b_x_step_size < 0:
                    # bounce in opposite direction (Player & Ball Collision)
                    self.b_x_step_size = -self.b_x_step_size
                    self.match_touches += 1  # update match touches

            if b.colliderect(p2):
                if self.b_x_step_size > 0:
                    # bounce in opposite direction (Player & Ball Collision)
                    self.b_x_step_size = -self.b_x_step_size
                    self.match_touches += 1  # update match touches

            # Update match time
            self.match_time += clock.tick(self.FRAME_RATE)

        # Game over (all matches finished): Score
        print("Game over")
        print("P1 Score: {}".format(self.game_p1_score))
        print("P2 Score: {}".format(self.game_p2_score))

    # A match is finished. Reset match statistics to prep for the next one
    def reset(self):

        # Reset ball x coordinate
        self.b_x = self.B_STARTX

        # Reset ball y coordinate (randomly centered)
        self.b_y = np.random.choice(
            range(round((self.SCREEN_GROUND - self.B_RADIUS) * 0.2), round((self.SCREEN_GROUND - self.B_RADIUS) * 0.8)))

        # Reset ball x velocity (random direction)
        self.b_x_step_size = self.INITIAL_B_X_STEP_SIZE * \
            np.random.choice([-1, 1])

        # Reset ball y velocity (random direction)
        self.b_y_step_size = self.INITIAL_B_Y_STEP_SIZE * \
            np.random.choice([-1, 1])

        # Reset Player 1 paddle x coordinate
        self.P1_X = self.P1_STARTX

        # Reset Player 1 paddle y coordinate
        self.p1_y = self.P1_STARTY

        # Reset Player 2 paddle x coordinate
        self.P2_X = self.P2_STARTX

        # Reset Player 2 paddle y coordinate
        self.p2_y = self.P2_STARTY

        # Save last match time
        self.last_match_time = self.match_time

        # Reset match time
        self.match_time = 0

        # Save last match touches
        self.last_match_touches = self.match_touches

        # Reset match touches
        self.match_touches = 0

        # Gap between matches
        sleep(0.2)
