# test_spike_encoding.py
import torch
import snntorch as spikegen
from snntorch import spikegen as sg

#  Fake a "frame" — normally this comes from VizDoom, but for testing
#    we just make up pixel brightness values between 0 and 1
fake_pixels = torch.tensor([0.9, 0.1, 0.5, 0.7])
print("Original pixel brightness values:", fake_pixels)

#  Convert to spikes over 10 time steps using rate coding
num_steps = 10
spike_train = sg.rate(fake_pixels, num_steps=num_steps)

print("Spike train shape:", spike_train.shape)   # should be (10, 4) -> 10 time steps, 4 pixels
print(spike_train)

#  Count how many times each pixel fired
spike_counts = spike_train.sum(dim=0)
print("Spike counts per pixel:", spike_counts)
print("(Compare to original brightness — higher brightness should fire more)")