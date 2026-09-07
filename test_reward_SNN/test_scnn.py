# test_scnn.py
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# 1. Define the network architecture
class SpikingDoomNet(nn.Module):
    def __init__(self, num_actions=3):
        super().__init__()

        beta = 0.9  # decay rate — same beta from your notes (LIF equation)
        spike_grad = surrogate.fast_sigmoid()  # surrogate gradient trick from Part 5.2 of your notes

        # Convolutional layer: scans the image for visual features (edges, shapes)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)  # LIF neuron layer, replaces ReLU

        self.pool = nn.MaxPool2d(2)  # shrinks the image, keeps important features

        # Second conv layer: combines simple features into more complex ones
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Flatten + dense layer: turns visual features into an action decision
        self.fc1 = nn.Linear(16 * 27 * 37, num_actions)  # 15,984 -> matches conv output, verified below
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps):
        # Initialize membrane potentials to zero — U[0] = 0, matching your notes' worked example
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        action_spikes_over_time = []

        for t in range(num_steps):
            cur1 = self.pool(self.conv1(x[t]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            action_spikes_over_time.append(spk3)

        # Sum spikes across time — action with the most spikes "wins" (like rate coding, in reverse)
        return torch.stack(action_spikes_over_time).sum(dim=0)


# 2. Quick sanity check: verify the flattened conv output size BEFORE building the real input
dummy = torch.rand(1, 1, 120, 160)
x = nn.MaxPool2d(2)(nn.Conv2d(1, 8, 5)(dummy))
x = nn.MaxPool2d(2)(nn.Conv2d(8, 16, 5)(x))
print("Flattened conv output shape (for sizing fc1):", x.shape)

# 3. Build the actual test input — shape (time_steps, batch, channels, height, width)
num_steps = 10
fake_input = torch.rand(num_steps, 1, 1, 120, 160)  # random values 0-1, pretend they're spikes

# 4. Run the network
net = SpikingDoomNet(num_actions=3)
output = net(fake_input, num_steps)

# Add this after computing 'output', before printing results

# Check if ANY neuron in the hidden layers ever fired, at any time step
with torch.no_grad():
    mem1 = net.lif1.init_leaky()
    mem2 = net.lif2.init_leaky()
    mem3 = net.lif3.init_leaky()
    total_spk1, total_spk2, total_spk3 = 0, 0, 0

    for t in range(num_steps):
        cur1 = net.pool(net.conv1(fake_input[t]))
        spk1, mem1 = net.lif1(cur1, mem1)
        cur2 = net.pool(net.conv2(spk1))
        spk2, mem2 = net.lif2(cur2, mem2)
        cur3 = net.fc1(spk2.flatten(1))
        spk3, mem3 = net.lif3(cur3, mem3)

        total_spk1 += spk1.sum().item()
        total_spk2 += spk2.sum().item()
        total_spk3 += spk3.sum().item()

    print(f"Layer 1 total spikes across all steps: {total_spk1}")
    print(f"Layer 2 total spikes across all steps: {total_spk2}")
    print(f"Layer 3 (output) total spikes across all steps: {total_spk3}")


print("Output shape:", output.shape)               # should be (1, 3) -> 1 batch, 3 possible actions
print("Action spike counts:", output)
print("Chosen action (most spikes):", output.argmax(dim=1).item())