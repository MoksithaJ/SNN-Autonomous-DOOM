import vizdoom as vzd
from time import sleep
import random

print("Initializing DOOM...")

game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.init()

print("DOOM started! Running some random actions...")

actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

#  run 3 full episodes instead of one flat loop of 100 steps
for episode in range(3):
    game.new_episode()          #  resets the map, monster, and score to start fresh
    total_reward = 0            #  we'll add up every reward this episode

    while not game.is_episode_finished():   # stop when the episode actually ends
        action = random.choice(actions)
        reward = game.make_action(action)   # reward is a number, usually 0 most steps
        total_reward += reward
        sleep(0.02)

    print(f"Episode {episode + 1} finished. Total reward: {total_reward}")

print("Finished! Closing DOOM.")
game.close()