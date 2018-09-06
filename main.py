import argparse
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time
import re
import csv

from game import Game
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from util import TILES, flatten, normalize, normalize_num, get_reward

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='specify if training is enabled', action='store_true')
parser.add_argument('-l', '--log', help='specify if extra logging is enabled', action='store_true')
parser.add_argument('-e', '--env', help='specify web or local env')
args = parser.parse_args()

CHROMEDRIVER_DIR = "ql3/bin/chromedriver"
MODEL_PATH = os.getcwd() + "/abcd.ckpt"
EXPERIENCES_PATH = os.getcwd() + "/abcd.p"
LOCAL = 'local'
WEB = 'web'

driver = None
env = args.env
mode = {'train': False, 'log': False}
mode['train'] = args.train
mode['log'] = args.log

if env == WEB:
    driver = webdriver.Chrome(CHROMEDRIVER_DIR)

# retry_button_class = 'retry-button'
# keep_going_button_class = 'keep-playing-button'

def read_screen():
    """Reads the 2048 board.

    Reads screen by parsing the div classes containing the tiles.
    Much cleaner than read_screen_with_image().

    Returns:
        grid: a 4x4 list containing the 2048 grid
    """
    tile_container = driver.find_element_by_class_name('tile-container')
    attempts = 5
    divs = []
    while attempts > 0:
        attempts -= 1
        try:
            divs = tile_container.find_elements_by_tag_name('div')
            break
        except:
            print("stale element, trying again")
            
    grid_classes = []
    for div in divs:
        grid_classes.append(div.get_attribute('class').split(" "))

    grid = [[0, 0, 0, 0] for i in range (4)]

    tile_regex = '^tile\-[0-9]+$'
    position_regex = '^tile\-position\-.+$'

    for grid_class in grid_classes:
        tile_class = [x for x in grid_class if re.match(tile_regex, x)]
        position_class = [x for x in grid_class if re.match(position_regex, x)]

        if tile_class == [] or position_class == []:
            continue

        tile_value = int(tile_class[0].split("-")[-1])
        pos_split = position_class[0].split("-")
        col = int(pos_split[-2]) - 1
        row = int(pos_split[-1]) - 1

        grid[row][col] = tile_value

    return grid

def get_board(env, game=None):
    if env == LOCAL:
        return game.grid
    else:
        return read_screen()

def display_board(board):
    for i in range(4):
        print(board[i])

def network():
    inpt = tf.placeholder("float", [None, 4, 4])
    input_layer = tf.reshape(inpt, [-1, 4, 4, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=128,
        kernel_size=[2, 2],
        padding="valid",
        activation=tf.nn.elu)

    #conv1 = tf.layers.dropout(inputs=conv1, rate=0.5, training=True)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=128,
        kernel_size=[2, 2],
        padding="valid",
        activation=tf.nn.elu)

    #conv2 = tf.layers.dropout(inputs=conv2, rate=0.5, training=True)

    convoutlayer = tf.reshape(conv2, [-1, 4 * 128])

    dense1 = tf.layers.dense(inputs=convoutlayer, units=256, activation=tf.nn.relu)
    # dense1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=True)
    dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
    # dense2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=True)
    dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu)
    # dense3 = tf.layers.dropout(inputs=dense3, rate=0.5, training=True)

    denseV = tf.layers.dense(inputs=dense3, units=256, activation=tf.nn.relu)
    # denseV = tf.layers.dropout(inputs=denseV, rate=0.5, training=True)
    denseV2 = tf.layers.dense(inputs=denseV, units=4, activation=tf.nn.relu)

    denseA = tf.layers.dense(inputs=dense3, units=256, activation=tf.nn.relu)
    # denseA = tf.layers.dropout(inputs=denseA, rate=0.5, training=True)
    denseA2 = tf.layers.dense(inputs=denseA, units=1, activation=tf.nn.relu)

    mergelayer = tf.concat([denseV2,denseA2], -1)
    
    logits = tf.layers.dense(inputs=mergelayer, units=4)

    return inpt, logits

def init():
    driver.delete_all_cookies()
    driver.get("http://gabrielecirulli.github.io/2048/")

    #disable animation
    with open("without-animation.js", "r") as myfile: 
        data = myfile.read().replace('\n', '')
    driver.execute_script(data)

def train(inpt, out, env, sess, mode):
    # Training parameters
    batch_size = 256
    discount = 0.9
    epochs = 500000000
    epsilon = 1
    learning_rate = 1e-5
    start = 0
    num_explore = 100000
    max_exps = 30000

    # Loss Function
    y = tf.placeholder("float", [None, 4])
    loss = tf.reduce_sum(tf.square(y - out))
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()
    if os.path.isfile(MODEL_PATH + '.data-00000-of-00001'):
        print("Loading model.")
        saver.restore(sess, MODEL_PATH)
        start = num_explore + max_exps
    else:
        print("Initializing new model.")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    # Experience Replay Memory
    data = {}
    experiences = []
    avg_scores = []
    epochs_trained = 0
    index = 0
    bests = [0, 0, 0, 0, 0, 0, 0]
    if os.path.isfile(EXPERIENCES_PATH):
        data = pickle.load(open(EXPERIENCES_PATH, 'rb'))
        experiences = data['experiences']
        epochs_trained = data['epochs_trained']
        time.sleep(10)
        avg_scores = data['avg_scores']
        max_exps = len(experiences)
    else:
        data['experiences'] = []
        data['epochs_trained'] = 0
        data['avg_scores'] = []

    # Setup
    game = None
    if env == LOCAL:
        game = Game()
        game.add_random_tile()

    score = 0
    board = get_board(env, game)
    inpt_board = flatten(normalize(board))
    prev_inpt_board = None
    prev_action = -1

    tot_score_counter = 0
    game_counter = 0
    step_counter = 0
    avg_score = 0
    game_step = 0 
    games_played = 0
    if mode['log']:
        csvfile = open("log.csv", "w")
        log_file = csv.writer(csvfile)
    for c in range (start, epochs):
        if env == WEB:
            game = driver.find_element_by_tag_name("body")
        feed_dict = {inpt: inpt_board.reshape(1,4,4)}

        action_indices = []
        action = -1

        # Epsilon-greedy action selection
        #epsilon = max(0 - float(games_played) / num_explore, 0.00)
        rand = random.random()
        #shift = 0  if game_step < 250 else 0.0 / np.log(np.log(games_played))
        if rand < 0:
            action_indices = [0,1,2,3]
            random.shuffle(action_indices)
        else:
            values = out.eval(feed_dict=feed_dict, session=sess)
            if env == WEB:
                print(values)
            action_indices = sorted([0, 1, 2, 3], key=lambda i: values[0][i], reverse=True)

        action, new_board = move(env, game, action_indices, board)
        game_step += 1
        
        # if none of the actions are valid, restart the game
        if action == -1:
            step_counter += game_step
            games_played += 1
            if env != WEB:
                max_tile = max(x for row in game.grid for x in row)
            if max_tile >= 2048:
                print(game.grid)
            if mode['log']:
                log_file.writerow([score, max_tile, game_step])
            if prev_inpt_board is not None:
                one_hot_reward = np.zeros((4))
                one_hot_reward[prev_action] = -1
                experience = (prev_inpt_board, one_hot_reward, inpt_board, prev_action)
                if max_exps > len(experiences):
                    experiences.append(experience)
                else:
                    experiences[index] = experience
                    index += 1
                    if index >= max_exps:
                        index = 0

            if env == LOCAL:
                #game.display()
                game = Game()
                game.add_random_tile()
            else:
                driver.find_element_by_class_name("restart-button").click()
                #time.sleep(0.5)
            
            #print "Final Score: %d" % score
            game_step = 0
            if len(experiences) == max_exps:
                tot_score_counter += score
                if score >= 12000:
                    bests[0] += 1
                elif score >= 10000:
                    bests[1] += 1
                elif score >= 8000:
                    bests[2] += 1
                elif score >= 6000:
                    bests[3] += 1
                elif score >= 4000:
                    bests[4] += 1
                elif score >= 2000:
                    bests[5] += 1
                elif score >= 0:
                    bests[6] += 1
                
                game_counter += 1
                if game_counter == 64:
                    print("Average score over last %d games: %d" % (game_counter, tot_score_counter/game_counter))
                    print("Average game length over last %d games: %d" % (game_counter, step_counter / game_counter))
                    print(bests)
                    print("Games played: %d" % games_played)
                    step_counter = 0
                    bests = [0, 0, 0, 0, 0, 0, 0]
                    # avg_scores.append(tot_score_counter/game_counter)
                    # data['experiences'] = experiences
                    # data['epochs_trained'] = c
                    # data['avg_scores'] = avg_scores
                    # pickle.dump(data, open(EXPERIENCES_PATH, 'wb'))
                    # print "Data saved!"
                    
                    game_counter = 0
                    tot_score_counter = 0

            score = 0
            board = get_board(env, game)
            inpt_board = flatten(normalize(board))
            prev_inpt_board = None
            continue

        if env == WEB:
            score_text = driver.find_element_by_class_name('score-container').text
            new_score = int(score_text.split("\n")[0])
        else:
            new_score = game.score

        reward = get_reward(new_score - score)
        score = new_score
        next_inpt_board = flatten(normalize(new_board))

        one_hot_reward = np.zeros((4))
        one_hot_reward[action] = reward
        experience = (inpt_board, one_hot_reward, next_inpt_board, action)

        if max_exps > len(experiences):
            experiences.append(experience)
        else:
            experiences[index] = experience
            index += 1
            if index >= max_exps:
                index = 0

        # Update board
        prev_action = action
        board = new_board
        prev_inpt_board = inpt_board
        inpt_board = next_inpt_board
        
        # Train the model using experience replay
        if c > 0 and c % batch_size == 0 and len(experiences) == max_exps and mode['train']:
            sample = random.sample(experiences, batch_size)
            next_state = np.array([x[2] for x in sample])
            rewards = np.array([x[1] for x in sample])
            cur_state = np.array([x[0] for x in sample])
            actions = np.array([x[3] for x in sample])

            rewards = np.array([reward[action] for reward, action in zip(rewards, actions)])
            values = out.eval(feed_dict={inpt: next_state}, session=sess)
            values = np.array([max(value) for value in values])
            ys = np.array([reward + discount * val for reward, val in zip(rewards, values)])
            cur_values = out.eval(feed_dict={inpt: cur_state}, session=sess)
            cur_values = np.array([np.array([x if i == action else val[i] for i in range(4)])  for val, action, x in zip(cur_values, actions, ys)])
            sess.run([train_opt],feed_dict = {
                inpt: cur_state,
                y: cur_values})


            if c % (batch_size * 1000) == 0:
                print("Saving model!")
                save_path = saver.save(sess, MODEL_PATH)


def move(env, game, action_indices, board):
    for i in range (4):
        moved = False
        action = action_indices[i]
        if env == LOCAL:
            moved = game.move(action)
        else:   
            if action == 0:
                game.send_keys(Keys.ARROW_UP)
            elif action == 1:
                game.send_keys(Keys.ARROW_RIGHT)
            elif action == 2:
                game.send_keys(Keys.ARROW_DOWN)
            elif action == 3:
                game.send_keys(Keys.ARROW_LEFT)
            time.sleep(.1)
            new_board = get_board(env, game)
            moved = new_board != board

        if moved:
            if env == LOCAL:
                game.add_random_tile()
            
            new_board = get_board(env, game)
            return [action, new_board]

    return [-1, None]


# main
random.seed(19980923)
sess = tf.Session()

if env == WEB:
    init()
    restart = driver.find_element_by_class_name("restart-button")
    restart.click()
    time.sleep(0.5) # allow game to load

inpt, out = network()
train(inpt, out, env, sess, mode)

if env == WEB:
    driver.quit()
