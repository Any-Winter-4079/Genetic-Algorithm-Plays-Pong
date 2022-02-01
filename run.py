#################
####   RUN   ####
#################

import sys
import pygame
import numpy as np
from time import sleep
from threading import Thread

from ga import *
from nn import *
from pong import *
from config import *


def max_function(genes):
    # collection of scores (points or time) AI scores per match Vs itself
    scores = []
    matches_to_play = n_train_matches if sys.argv[1] == "train" else n_play_matches
    while len(scores) < matches_to_play:
        count = pong.match_count
        p1_choice_number = 0
        p2_choice_number = 0
        while count == pong.match_count:

            # Player 1 Features seen from the world
            input_features = np.array([
                abs((pong.b_x - pong.B_RADIUS) - (pong.P1_X + pong.P1_WIDTH)) /
                (pong.SCREEN_WIDTH - pong.P1_WIDTH - pong.P2_WIDTH),
                abs((pong.b_y) - (pong.p1_y + pong.P1_HEIGHT // 2)) /
                (pong.SCREEN_GROUND - pong.P1_HEIGHT),
                1 if (pong.b_y) >= (pong.p1_y + pong.P1_HEIGHT // 2) else -1
            ])
            # print("P1: {}".format(input_features))

            # Player 1 Prediction
            prediction = forward(input_features, genes)
            # print("P1 Predictions: {}".format(prediction))
            p1_choice_number += 1
            choice = max(prediction)
            if prediction[0] == choice:
                pong.p1_offset = pong.P1_STEP_SIZE  # Move down
                # print("P1 Choice {}: move down".format(p1_choice_number))
            elif prediction[2] == choice:
                pong.p1_offset = -pong.P1_STEP_SIZE  # Move up
                # print("P1 Choice {}: move up".format(p1_choice_number))
            else:
                pong.p1_offset = 0  # Stay put
                # print("P1 Choice {}: stay put".format(p1_choice_number))

            # If Train mode
            if sys.argv[1] == "train":

                # Player 2 Features seen from the world
                input_features = np.array([
                    abs((pong.b_x + pong.B_RADIUS) - (pong.P2_X)) /
                    (pong.SCREEN_WIDTH - pong.P1_WIDTH - pong.P2_WIDTH),
                    abs((pong.b_y) - (pong.p2_y + pong.P2_HEIGHT // 2)) /
                    (pong.SCREEN_GROUND - pong.P2_HEIGHT),
                    1 if (pong.b_y) >= (pong.p2_y +
                                        pong.P2_HEIGHT // 2) else -1
                ])
                # print("P2: {}".format(input_features))

                # Player 2 Prediction
                prediction = forward(input_features, genes)
                # print("P2 Predictions: {}".format(prediction))
                p2_choice_number += 1
                choice = max(prediction)
                if prediction[0] == choice:
                    pong.p2_offset = pong.P2_STEP_SIZE  # Move down
                    # print("P2 Choice {}: move down".format(p2_choice_number))
                elif prediction[2] == choice:
                    pong.p2_offset = -pong.P2_STEP_SIZE  # Move up
                    # print("P2 Choice {}: move up".format(p2_choice_number))
                else:
                    pong.p2_offset = 0  # Stay put
                    # print("P2 Choice {}: stay put".format(p2_choice_number))

            # Sleep
            sleep(pong.FRAME_RATE / 1000)

            # Check max time, to avoid infinite playing
            if pong.match_time > 30*1000:
                pong.reset()
                pong.match_count += 1
                sleep(0.1)

        # Save touches
        # touches = pong.last_match_touches
        _time = pong.last_match_time / 1000
        scores.append(_time)
        print("Score: {}".format(_time))

    # Obtain average score
    final_score = sum(scores) / len(scores)
    print("Average Game Score: {}".format(final_score))
    print()
    sys.stdout.flush()

    return -final_score  # max function, GA works with min, so we * -1


def run_AI(game_type):
    if game_type == "train":
        myGA = GA(max_function, bounds, p_size)
        myGA.run(n_generations, dtype="discrete", debug=True)
        bestGenome = myGA.best()
        print("Best: {}".format(bestGenome))
    else:
        genes = np.array(
            [6, 5, -3, 6, -4, -6, -2, 3, 9, -6, -8, 8, 6, -1, -10, 9, -3, 6])
        print("Genes: {}".format(genes))
        max_function(genes)


def init(game_type):
    global pong
    thread = Thread(target=run_AI, args=(game_type, ))
    # Start Pong (black screen)
    pong = Pong(FRAME_RATE,
                SCREEN_BG_PATH, SCREEN_START, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_GROUND,
                FONT_PATH, FONT_SIZE, FONT_COLOR,
                SCORE1_POS, SCORE2_POS,
                P1_IMAGE_PATH, P1_WIDTH, P1_HEIGHT, P1_STEP_SIZE, P1_X, p1_y,
                P2_IMAGE_PATH, P2_WIDTH, P2_HEIGHT, P2_STEP_SIZE, P2_X, p2_y,
                B_COLOR, B_RADIUS, b_x_step_size, B_X_MAX_STEP_SIZE, b_y_step_size, B_Y_MAX_STEP_SIZE, b_x, b_y)
    # Then start AI to control paddle(s)
    thread.start()
    # Run game (render players and start moving)
    if game_type == "train":
        pong.run(n_generations=n_generations,
                 p_size=p_size, n_train_matches=n_train_matches)
    else:
        pong.run(n_play_matches=n_play_matches)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        # Wrong arguments
        print("Use: python3 run.py train")
        print("Use: python3 run.py play")
    else:
        # Init game
        pong = None
        init(sys.argv[1])
