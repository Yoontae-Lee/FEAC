import torch
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from scipy.special import exp1
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
NUM_CLIENTS = 200
COMMUNICATION_ROUNDS = 1000
BATCH_SIZE = 128
EPOCHS = 1
LR = 0.01
CLIENT_FRACTION = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAXIMUM_DISTANCE = 100
CELL_IN_DISTANCE = 100
SUB_CHANNELS = 1
TRANS_POWER_PER_DEVICE = 0.1
DOWNLINK_POWER = 0.1
GAUSSIAN_NOISE = 10**(-11)
NON_TRUNC_PROBABILITY = 0.4

# MnistNet model 
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Local training
def train_local_model(model, dataloader, criterion, num_epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    accumulated_gradients = [torch.zeros_like(param, device=DEVICE) for param in model.parameters()]
    for epoch in range(num_epochs):  # Iterate over multiple epochs
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        accumulated_gradients[i] += param.grad
    return accumulated_gradients

# Evaluation function 
def evaluate(model, dataloader, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss

# Flatten & Unflatten
def flatten(params):
    return torch.cat([param.view(-1) for param in params])

def unflatten(flat_param, ref_list):
    unflat = []
    idx = 0
    for param in ref_list:
        param_shape = param.shape
        num_elements = param.numel()
        unflat.append(flat_param[idx : idx + num_elements].view(param_shape))
        idx += num_elements
    return unflat

def gradient_quantization(gradients):
    return torch.sign(gradients)

def gradients_aggregation(gradients):
    sum_gradients = torch.zeros_like(gradients[0])
    for gradient in gradients:
        sum_gradients += gradient
    return sum_gradients

# Wireless transmission functions 
def device_distance(r_k, Rin):
    if r_k <= Rin:
        return r_k
    return 0

# Complex Gaussian generation
def complex_gaussian():
    real_part = np.random.normal(0, np.sqrt(0.5))
    imag_part = np.random.normal(0, np.sqrt(0.5))
    h_k_m = real_part + 1j * imag_part
    return h_k_m

# Receive SNR
def receive_snr(P0, M, g_th):
# def receive_snr(g_th):
    ei_value = exp1(g_th)
    rho_0 = P0 / (M * ei_value)
    # rho_0 = AVG_SNR / ei_value
    return rho_0

# Power allocation with truncated channel inversion
def trun_channel_inversion(h_k_m, g_th, rho_0):
    channel_gain = (np.abs(h_k_m))**2
    if channel_gain >= g_th:
        return (np.sqrt(rho_0) * np.conj(h_k_m)) / channel_gain
    return 0

# Gradient to signal transmission (BPSK)
def param_transmission(h_k_m, p_k_m, gradient):
    h_real = np.real(h_k_m)
    p_real = np.real(p_k_m)
    transmission = (h_real * p_real * gradient).clone().detach().to(DEVICE)
    return transmission.to(DEVICE)

################################################### Downlink ###################################################
def downlink_transmission(global_model, P_dl):
    global_model_flatten  = flatten(global_model.parameters())
    # L2 normalizatio of globla model
    norm_model = torch.linalg.norm(global_model_flatten) ** 2  
    # Calculating scaling factor alpha downlink
    alpha_dl = torch.sqrt(P_dl / norm_model)  
    # Nosie 
    h_dl = {m: complex_gaussian() for m in range(NUM_CLIENTS)}
    # Received siganl from PS to each clients
    y_dl = {}
    noise_dl = torch.sqrt(torch.tensor(GAUSSIAN_NOISE)).to(DEVICE)
    for m in range(NUM_CLIENTS):
        y_dl_m = alpha_dl * h_dl[m] * global_model_flatten + torch.normal(0, noise_dl, size=global_model_flatten.shape).to(DEVICE)
        y_dl[m] = y_dl_m  
    return y_dl, h_dl, alpha_dl

################################################## Downlink End ################################################
################################################################################################################

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Split data among clients
split_data_sizes = [len(dataset) // NUM_CLIENTS] * NUM_CLIENTS
split_data_sizes[-1] += len(dataset) % NUM_CLIENTS
client_datasets = random_split(dataset, split_data_sizes)

global_model = MnistNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

accuracy_list = []
loss_list = []

for round in range(COMMUNICATION_ROUNDS):
    print(f"Round {round + 1}/{COMMUNICATION_ROUNDS}")
    
    # Sample clients
    sampled_clients = random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * CLIENT_FRACTION))

    # Lists and dictionaries for tracking
    participating_clients = []
    client_gradient = []
    client_distances = {}
    
    # Downlink  transmission
    y_dl, h_dl, alpha_dl = downlink_transmission(global_model, DOWNLINK_POWER)

    # Check distance for each sampled client
    for client_id in sampled_clients:
        # Generate random distance for the client
        r_k_m = random.randint(10, MAXIMUM_DISTANCE)
        r_k_interior = device_distance(r_k_m, CELL_IN_DISTANCE)
        
        # Only proceed if device is within cell interior distance
        if r_k_interior > 0:
            participating_clients.append(client_id)
            client_distances[client_id] = r_k_interior  # Store the distance

            # Model recovery from downlink
            h_dl_m = h_dl[client_id] 
            recieved_model = y_dl[client_id]
            est_model = torch.real(recieved_model / (alpha_dl*h_dl_m))
            est_model_unflatten = unflatten(est_model, global_model.parameters())
            
            # Global model to local model
            local_model = MnistNet().to(DEVICE)
            state_dict = {name: param for (name, _), param in zip(local_model.state_dict().items(), est_model_unflatten)}
            local_model.load_state_dict(state_dict)

            # Train local model with τ iterations
            local_gradient = train_local_model(local_model, client_loaders[client_id], criterion, EPOCHS)
            client_gradient.append((client_id, local_gradient))
    
    # Log participating clients
    print(f"Participating Devices in Round {round + 1} (within {CELL_IN_DISTANCE}m): {participating_clients}")
    print(f"Number of participating devices: {len(participating_clients)}/{len(sampled_clients)}")
    
    # Only proceed with aggregation if there are participating clients
    if participating_clients:
        # Wireless transmission simulation
        below_gth = 0
        signal_gradients = []
        for client_id, gradient in client_gradient:
            quant_gradients = [gradient_quantization(grad) for grad in gradient]
            flatten_gradient = flatten(quant_gradients)
            
            h_k_m = complex_gaussian()
            g_th = -np.log(NON_TRUNC_PROBABILITY)
            # rho_0 = receive_snr(g_th)
            rho_0 = receive_snr(TRANS_POWER_PER_DEVICE, SUB_CHANNELS, g_th)
            p_k_m = trun_channel_inversion(h_k_m, g_th, rho_0)
            if p_k_m == 0:
                below_gth += 1
                continue  
            signal_state_dict = param_transmission(h_k_m, p_k_m, flatten_gradient)
            signal_gradients.append(signal_state_dict)

        print(f"Devices below g_th: {below_gth}/{len(participating_clients)}")
        
        # Aggregate gradients
        global_gradients = gradients_aggregation(signal_gradients)
        
        # Add channel noise
        std_noise = torch.sqrt(torch.tensor(GAUSSIAN_NOISE)).to(DEVICE)
        global_noise = global_gradients + torch.normal(mean=0.0, std=std_noise, size=global_gradients.shape).to(DEVICE)
        
        # Process aggregated gradients
        unflatten_gradient = unflatten(global_noise, quant_gradients)

        # Apply Majority Voting
        majority_voting = [torch.sign(u_g) for u_g in unflatten_gradient]
        
        # Update global model
        optimizer = optim.SGD(global_model.parameters(), lr=LR, momentum=0.9)
        optimizer.zero_grad()
        for param, grad in zip(global_model.parameters(), majority_voting):
            param.grad = grad
        optimizer.step()
        
        # Evaluate global model
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        accuracy, avg_loss = evaluate(global_model, test_loader, criterion)
        
        accuracy_list.append(accuracy)
        loss_list.append(avg_loss)
        
        print(f"Global Model Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    else:
        print("No devices within cell interior distance. Skipping round.")

print("Federated Learning Complete.")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, linestyle='-', linewidth=0.4) 
plt.xlabel('Communication Round')
plt.ylabel('Accuracy')
plt.title('Global Model Accuracy Over Rounds')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(loss_list) + 1), loss_list, linestyle='-', color='red', linewidth=0.4) 
plt.xlabel('Communication Round')
plt.ylabel('Average Loss')
plt.title('Global Model Loss Over Rounds')

plt.tight_layout()
# Save
plt.savefig("obda_04.png")  
print("Plot saved as obda_04.png")

# Save into excel
df = pd.DataFrame({
    'Round': list(range(1, len(accuracy_list) + 1)),
    'Accuracy': accuracy_list,
    'Loss': loss_list
})
df.to_excel("obda_04.xlsx", index=False)

print("Results saved to obda_04.xlsx")

