# test_frame_to_spikes.py
import vizdoom as vzd
import torch
from snntorch import spikegen as sg
import numpy as np

game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.set_screen_format(vzd.ScreenFormat.GRAY8)   # NEW: ask VizDoom for grayscale directly
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)  # NEW: small resolution
game.init()

game.new_episode()
state = game.get_state()
frame = state.screen_buffer   # shape: (120, 160) - grayscale, no color channel

print("Raw frame shape:", frame.shape)
print("Raw pixel value range:", frame.min(), "to", frame.max())

#  Normalize pixels from [0, 255] to [0, 1] — spike encoding needs values in this range
frame_tensor = torch.tensor(frame, dtype=torch.float32) / 255.0

#  Convert the WHOLE frame (120x160 pixels) to spikes over 10 time steps
num_steps = 10
spike_frame = sg.rate(frame_tensor, num_steps=num_steps)

print("Spike frame shape:", spike_frame.shape)  # should be (10, 120, 160)
print("Total spikes across whole frame, per time step:", spike_frame.sum(dim=(1,2)))
print("Mean pixel brightness (0-1):", frame_tensor.mean().item())
game.close()