############################
####  NN CONFIGURATION  ####
############################

# Features seen from the world
n_input_features = 3
n_hidden_units = 3  # 1 hidden layer with n neurons
# Predictions
# Move down probability, Stay Put Probability, Move up Probability
n_output_features = 3

############################
####  GA CONFIGURATION  ####
############################

n_generations = 10   # How many generations to train our AI for
p_size = 10  # How many individuals each population (generation) has
n_genes = n_input_features * n_hidden_units + \
    n_hidden_units * n_output_features  # genes per individual
bounds = [(-10, 10)] * n_genes  # bounds for each gene (var) to learn

##############################
####  PONG CONFIGURATION  ####
##############################

# Matches on train mode
# How many matches an individual (AI) plays Vs itself (AI) to obtain avg fitness
n_train_matches = 1
# Games on play mode
n_play_matches = 25  # How many matches AI plays Vs YOU

# Pygame
FRAME_RATE = 60

# Screen
SCREEN_BG_PATH = "Images/bg.png"
SCREEN_START = 0
SCREEN_WIDTH = 750  # BG width
SCREEN_HEIGHT = 500  # BG height
SCREEN_GROUND = SCREEN_HEIGHT - 109  # Pixels from top to bounceable ground

# Font
FONT_PATH = "Fonts/PressStart2P-Regular.ttf"
FONT_SIZE = 24
FONT_COLOR = (255, 255, 255)

# Score
SCORE1_POS = (60, 20)
SCORE2_POS = (SCREEN_WIDTH - 125, 20)

# Player 1: AI
P1_IMAGE_PATH = "Images/short-paddle.png"
P1_WIDTH = 20     # Paddle width
# P1_HEIGHT = 36  # Tiny Paddle height
P1_HEIGHT = 68  # Short Paddle height
# P1_HEIGHT = 134   # Normal Paddle height
P1_STEP_SIZE = 4  # How much AI can move (up | down) per time unit
P1_X = SCREEN_START  # Fixed x position (left of Player 1's paddle)
# Initial y position (p1_y at top of Player 1's paddle)
p1_y = (SCREEN_GROUND - P1_HEIGHT) // 2

# Player 2: YOU | AI
P2_IMAGE_PATH = "Images/short-paddle.png"
P2_WIDTH = 20     # Paddle width
# P2_HEIGHT = 36  # Tiny Paddle height
P2_HEIGHT = 68  # Short Paddle height
# P2_HEIGHT = 134   # Normal Paddle height
P2_STEP_SIZE = 4  # How much YOU | AI can move (up | down) per time unit
P2_X = SCREEN_WIDTH - P2_WIDTH  # Fixed x position (left of Player 2's paddle)
# Initial y position (p2_y at top of Player 2 paddle)
p2_y = (SCREEN_GROUND - P1_HEIGHT) // 2

# Ball
B_COLOR = (255, 223, 160)
B_RADIUS = 10
b_x_step_size = 4  # How far the ball moves horizontally per time unit initially
B_X_MAX_STEP_SIZE = 4  # How far the ball moves horizontally per time unit at most
b_y_step_size = 4  # How far the ball moves vertically per time unit initially
B_Y_MAX_STEP_SIZE = 4  # How far the ball moves vertically per time unit at most
b_x = SCREEN_WIDTH // 2  # Initial x position (b_x at ball's center)
b_y = SCREEN_GROUND // 2  # Initial y position (b_y at ball's center)
