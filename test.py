import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from train_new import SkipFrame, NormalizeObservation

import cv2
import numpy as np
import torch
import random
from student_agent import Agent
import imageio
from IPython.display import Image
import matplotlib.pyplot as plt

test_mode = 1

def test(env, agent=None, gif_path="debug.gif"):
    frames = [] # Uncomment to store the frames for the animation

    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    rewards = []

    while not done:
        if test_mode == 1:
            action = agent.act(state)
        else:
            action = agent.act2(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        # Uncomment to store the frames for the animation
        frame = env.render(mode="rgb_array")
        frames.append(frame.copy()) 
        rewards.append(total_reward)
        print(f"\rStep: {step}, Reward: {total_reward}", end="", flush=True)

    print(f"\rStep: {step}, Reward: {total_reward}")

    env.close()
    plt.plot(rewards)
    plt.title("Total reward history")
    plt.xlabel("Step")
    plt.ylabel("Total reward")
    plt.savefig("reward_history.png")
    plt.close()

    # Uncomment to show the animation
    imageio.mimsave(gif_path, frames, duration=20)

if __name__=="__main__":

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    n_stack = 4
    n_skip = 4
    img_shape = (84, 84)
    if test_mode == 2:
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=img_shape)
        env = NormalizeObservation(env)
        env = SkipFrame(env, n_skip=n_skip)
        env = FrameStack(env, num_stack=n_stack)

    agent = Agent()
    test(env, agent, gif_path="debug.gif")
