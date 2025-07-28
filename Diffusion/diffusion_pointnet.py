import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#########################
# Device Setup
#########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#########################
# Functions
#########################

def plotSolution(x_coord,y_coord,solution,file_name,title):
    plt.scatter(x_coord, y_coord, s=2.5,c=solution,cmap='jet')
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    x_upper = np.max(x_coord) + 1
    x_lower = np.min(x_coord) - 1
    y_upper = np.max(y_coord) + 1
    y_lower = np.min(y_coord) - 1
    plt.xlim([x_lower, x_upper])
    plt.ylim([y_lower, y_upper])
    plt.gca().set_aspect('equal', adjustable='box')
    cbar= plt.colorbar()
    plt.savefig(file_name+'.png',dpi=300)
    #plt.savefig(file_name+'.eps')
    #plt.savefig(file_name+'.pdf',format='pdf')
    plt.clf()
    #plt.show()

def compute_rms_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    squared_diff = (generated - ground_truth) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rms_error = np.sqrt(mean_squared_diff)
    return rms_error

def compute_relative_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    difference = generated - ground_truth
    norm_difference = np.linalg.norm(difference)
    norm_ground_truth = np.linalg.norm(ground_truth)
    relative_error = norm_difference / norm_ground_truth
    return relative_error

#########################
# Data Loading & Preprocessing
#########################

# Load your data and extract point coordinates and CFD fields.
Data = np.load('/scratch/users/kashefi/DiffCFD/CFDdata.npy')
data_number = Data.shape[0]

point_numbers = 1024
space_variable = 2   # (x, y)
cfd_variable = 3     # (u, v, p)

input_data = np.zeros([data_number, point_numbers, space_variable], dtype='float32')
output_data = np.zeros([data_number, point_numbers, cfd_variable], dtype='float32')

for count in range(data_number):
    input_data[count, :, 0] = Data[count, :, 0]  # x coordinate
    input_data[count, :, 1] = Data[count, :, 1]  # y coordinate
    output_data[count, :, 0] = Data[count, :, 3]  # u
    output_data[count, :, 1] = Data[count, :, 4]  # v
    output_data[count, :, 2] = Data[count, :, 2]  # p

# Normalize input coordinates to [-1, 1]
x_min = np.min(input_data[:, :, 0])
x_max = np.max(input_data[:, :, 0])
y_min = np.min(input_data[:, :, 1])
y_max = np.max(input_data[:, :, 1])
input_data[:, :, 0] = 2 * (input_data[:, :, 0] - x_min) / (x_max - x_min) - 1
input_data[:, :, 1] = 2 * (input_data[:, :, 1] - y_min) / (y_max - y_min) - 1

# Split indices for training, validation, and testing
#all_indices = np.random.permutation(data_number)
#training_idx = all_indices[:int(0.8 * data_number)]
#validation_idx = all_indices[int(0.8 * data_number):int(0.9 * data_number)]
#test_idx = all_indices[int(0.9 * data_number):]

training_idx = np.load('training_idx.npy')
validation_idx = np.load('validation_idx.npy')
test_idx = np.load('test_idx.npy')

input_train = input_data[training_idx]
input_validation = input_data[validation_idx]
input_test = input_data[test_idx]

output_train = output_data[training_idx]
output_validation = output_data[validation_idx]
output_test = output_data[test_idx]

# Normalize the output (CFD fields) based on the training set
u_min = np.min(output_train[:, :, 0])
u_max = np.max(output_train[:, :, 0])
v_min = np.min(output_train[:, :, 1])
v_max = np.max(output_train[:, :, 1])
p_min = np.min(output_train[:, :, 2])
p_max = np.max(output_train[:, :, 2])

output_train[:, :, 0] = (output_train[:, :, 0] - u_min) / (u_max - u_min) 
output_train[:, :, 1] = (output_train[:, :, 1] - v_min) / (v_max - v_min)
output_train[:, :, 2] = (output_train[:, :, 2] - p_min) / (p_max - p_min)

output_validation[:, :, 0] = (output_validation[:, :, 0] - u_min) / (u_max - u_min)
output_validation[:, :, 1] = (output_validation[:, :, 1] - v_min) / (v_max - v_min) 
output_validation[:, :, 2] = (output_validation[:, :, 2] - p_min) / (p_max - p_min)

#########################
# Diffusion Setup
#########################

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas

T = 1000
betas = cosine_beta_schedule(T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

#########################
# Time Embedding
#########################

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        """
        Args:
          t: Tensor of shape [B] (timesteps as integers)
        Returns:
          Time embeddings of shape [B, emb_dim]
        """
        half_dim = self.emb_dim // 2
        emb_factor = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_factor)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

####################################
# PointNet-based Denoising Network
####################################
class PointNetMLP(nn.Module):
    def __init__(self, field_dim=3, coord_dim=2, time_emb_dim=32, scaling=2.0):
        super(PointNetMLP, self).__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        in_channels = field_dim + coord_dim + time_emb_dim

        # shared MLP (64, 64)
        self.conv1 = nn.Conv1d(in_channels, int(64 * scaling), 1)
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.conv2 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))

        # shared MLP (64, 128, 1024)
        self.conv3 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.conv4 = nn.Conv1d(int(64 * scaling), int(128 * scaling), 1)
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.conv5 = nn.Conv1d(int(128 * scaling), int(1024 * scaling), 1)
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))

        # shared MLP (512, 256, 128)
        self.conv6 = nn.Conv1d(int(1024 * scaling) + int(64 * scaling), int(512 * scaling), 1)
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.conv7 = nn.Conv1d(int(512 * scaling), int(256 * scaling), 1)
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.conv8 = nn.Conv1d(int(256 * scaling), int(128 * scaling), 1)
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))

        # shared MLP (128, output_channels)
        self.conv9 = nn.Conv1d(int(128 * scaling), int(128 * scaling), 1)
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))
        self.conv10 = nn.Conv1d(int(128 * scaling), field_dim, 1)

    def forward(self, noisy_field, coords, t):

        B, N, _ = noisy_field.shape
        t_emb = self.time_emb(t)                    
        t_emb_exp = t_emb.unsqueeze(1).repeat(1, N, 1)  
        x = torch.cat([noisy_field, coords, t_emb_exp], dim=-1)
        x = x.transpose(1, 2)
        
        # shared MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        local_feature = x

        # Shared MLP (64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, N)

        # concatenate local and global features
        x = torch.cat([local_feature, global_feature], dim=1)

        # shared MLP (512, 256, 128)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        # shared MLP (128, output_channels)
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.conv10(x)
        x = x.transpose(1, 2)
        return x

#########################
# Training & Validation Setup
#########################
model = PointNetMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 240000
batch_size = 256

# Prepare training dataset and loader.
train_dataset = TensorDataset(torch.tensor(output_train), torch.tensor(input_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare validation dataset and loader.
val_dataset = TensorDataset(torch.tensor(output_validation), torch.tensor(input_validation))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store loss evolution.
train_loss_history = []
val_loss_history = []

#########################
# Training Loop
#########################

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for clean_field, coords in train_loader:
        clean_field = clean_field.to(device)  # [B, N, 3]
        coords = coords.to(device)            # [B, N, 2]
        B = clean_field.shape[0]
        # Sample a random diffusion timestep for each sample in the batch.
        t = torch.randint(0, T, (B,), device=device).long()
        # Gather the corresponding alpha_bar values.
        alpha_bar_t = alpha_bars[t].view(B, 1, 1)
        # Sample noise from a standard normal distribution.
        noise = torch.randn_like(clean_field)
        # Compute the noisy field at timestep t.
        noisy_field = torch.sqrt(alpha_bar_t) * clean_field + torch.sqrt(1 - alpha_bar_t) * noise
        # Forward pass: predict the noise added.
        optimizer.zero_grad()
        pred_noise = model(noisy_field, coords, t)
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_train_loss = epoch_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    # Validation Loss Evaluation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clean_field, coords in val_loader:
            clean_field = clean_field.to(device)
            coords = coords.to(device)
            B = clean_field.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()
            alpha_bar_t = alpha_bars[t].view(B, 1, 1)
            noise = torch.randn_like(clean_field)
            noisy_field = torch.sqrt(alpha_bar_t) * clean_field + torch.sqrt(1 - alpha_bar_t) * noise
            pred_noise = model(noisy_field, coords, t)
            loss_val = F.mse_loss(pred_noise, noise)
            val_loss += loss_val.item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}", flush=True)

#########################
# Plot Loss Evolution
#########################

plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Loss Evolution')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('loss_evolution.png', dpi=300, bbox_inches='tight') 
plt.clf()
#plt.show()

#########################
# Sampling (Reverse Diffusion)
#########################

def sample(model, coords, num_steps=T):
    """
    Given a set of point coordinates, generate a sample of the CFD fields
    using the learned diffusion model.
    
    Args:
      model: the trained denoising network.
      coords: Tensor of shape [B, N, 2] with the input point coordinates.
      num_steps: number of diffusion steps (default T).
    
    Returns:
      x: Tensor of shape [B, N, 3] representing the denoised CFD fields.
    """
    model.eval()
    B, N, _ = coords.shape
    # Start from pure Gaussian noise.
    x = torch.randn((B, N, 3), device=device)
    # Reverse diffusion process.
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        beta = betas[t]
        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        with torch.no_grad():
            pred_noise = model(x, coords, t_tensor)
        # DDPM reverse update (simplified): ### why simplified?!
        x = 1 / torch.sqrt(alpha) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise
    return x


#########################
# Save Model
#########################

torch.save(model, 'full_model.pth')
print("Entire model saved to full_model.pth")


#########################
# Error Analysis using Diffusion Sampling (Training Set)
#########################
rms_u, rms_v, rms_p = 0.0, 0.0, 0.0
lrms_u, lrms_v, lrms_p = 0.0, 0.0, 0.0

# Loop over all samples in input_train (which is already the training subset)
for j in range(len(input_train)):
    curr_idx = j  # since input_train is already a subset, use its index directly
    
    # Convert the current training sample to a torch tensor (add batch dimension).
    input_data_pred = torch.tensor(input_train[curr_idx]).unsqueeze(0).float()  # Shape: [1, N, 2]
    
    with torch.no_grad():
        input_data_pred = input_data_pred.to(device)
        # Sample the CFD fields using the diffusion model.
        predictions = sample(model, input_data_pred)  # Shape: [1, N, 3] (normalized outputs)
        # Permute from [B, N, 3] to [B, 3, N]
        predictions = predictions.permute(0, 2, 1)
    
    # Denormalize coordinates for plotting
    input_train[j,:,0] = (input_train[j,:,0] + 1) * (x_max - x_min) / 2 + x_min
    input_train[j,:,1] = (input_train[j,:,1] + 1) * (y_max - y_min) / 2 + y_min

    predictions[0,0,:] = (predictions[0,0,:])*(u_max - u_min) + u_min
    predictions[0,1,:] = (predictions[0,1,:])*(v_max - v_min) + v_min
    predictions[0,2,:] = (predictions[0,2,:])*(p_max - p_min) + p_min

    output_train[j,:,0] = (output_train[j,:,0])*(u_max - u_min) + u_min
    output_train[j,:,1] = (output_train[j,:,1])*(v_max - v_min) + v_min
    output_train[j,:,2] = (output_train[j,:,2])*(p_max - p_min) + p_min
    
    #plot
    #plotSolution(input_train[j,:,0], input_train[j,:,1], predictions[0,0,:].cpu().numpy(),'u_pred_train'+str(j),'u')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], output_train[j,:,0],'u_truth_train'+str(j),'u')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], np.abs(predictions[0,0,:].cpu().numpy()-output_train[j,:,0]),'u_abs_train'+str(j),'u')

    #plotSolution(input_train[j,:,0], input_train[j,:,1], predictions[0,1,:].cpu().numpy(),'v_pred_train'+str(j),'v')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], output_train[j,:,1],'v_truth_train'+str(j),'v')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], np.abs(predictions[0,1,:].cpu().numpy()-output_train[j,:,1]),'v_abs_train'+str(j),'v')

    #plotSolution(input_train[j,:,0], input_train[j,:,1], predictions[0,2,:].cpu().numpy(),'p_pred_train'+str(j),'p')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], output_train[j,:,2],'p_truth_train'+str(j),'p')
    #plotSolution(input_train[j,:,0], input_train[j,:,1], np.abs(predictions[0,2,:].cpu().numpy()-output_train[j,:,2]),'p_abs_train'+str(j),'p')

    rms_u += compute_rms_error(predictions.cpu().numpy(), 0, output_train[j,:,0])
    lrms_u += compute_relative_error(predictions.cpu().numpy(), 0, output_train[j,:,0])

    rms_v += compute_rms_error(predictions.cpu().numpy(), 1, output_train[j,:,1])
    lrms_v += compute_relative_error(predictions.cpu().numpy(), 1, output_train[j,:,1])

    rms_p += compute_rms_error(predictions.cpu().numpy(), 2, output_train[j,:,2])
    lrms_p += compute_relative_error(predictions.cpu().numpy(), 2, output_train[j,:,2])

print()
print("Average RMS of Training for u: ", rms_u / len(training_idx))
print("Average Relative of Training for u: ", lrms_u / len(training_idx))
print()
print("Average RMS of Training for v: ", rms_v / len(training_idx))
print("Average Relative of Training for v: ", lrms_v / len(training_idx))
print()
print("Average RMS of Training for p: ", rms_p / len(training_idx))
print("Average Relative of Training for p: ", lrms_p / len(training_idx))

print()
print("############################################################")
print()
   
#########################################################

rms_u, rms_v, rms_p = 0.0, 0.0, 0.0
lrms_u, lrms_v, lrms_p = 0.0, 0.0, 0.0

u_collection = []
v_collection = []
p_collection = []

# Loop over all samples in input_test (which is already the testing subset)
for j in range(len(input_test)):
    curr_idx = j  # use the index within input_test directly
    
    # Convert the current test sample to a torch tensor (add batch dimension).
    input_data_pred = torch.tensor(input_test[curr_idx]).unsqueeze(0).float()  # Shape: [1, N, 2]
    
    with torch.no_grad():
        input_data_pred = input_data_pred.to(device)
        predictions = sample(model, input_data_pred)  # Output: [1, N, 3] (raw predictions)
        predictions = predictions.permute(0, 2, 1)       # Rearranged to [1, 3, N]
    
    input_test[j,:,0] = (input_test[j,:,0] + 1)*(x_max - x_min)/2 + x_min
    input_test[j,:,1] = (input_test[j,:,1] + 1)*(y_max - y_min)/2 + y_min

    predictions[0,0,:] = (predictions[0,0,:])*(u_max - u_min) + u_min
    predictions[0,1,:] = (predictions[0,1,:])*(v_max - v_min) + v_min
    predictions[0,2,:] = (predictions[0,2,:])*(p_max - p_min) + p_min

    #plot
    plotSolution(input_test[j,:,0], input_test[j,:,1], predictions[0,0,:].cpu().numpy(),'u_pred_test'+str(j),r'Prediction of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], output_test[j,:,0],'u_truth_test'+str(j),r'Ground truth of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], np.abs(predictions[0,0,:].cpu().numpy()-output_test[j,:,0]),'u_abs_test'+str(j),r'Absolute error of velocity $u$ (m/s)')

    plotSolution(input_test[j,:,0], input_test[j,:,1], predictions[0,1,:].cpu().numpy(),'v_pred_test'+str(j),r'Prediction of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], output_test[j,:,1],'v_truth_test'+str(j),r'Ground truth of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], np.abs(predictions[0,1,:].cpu().numpy()-output_test[j,:,1]),'v_abs_test'+str(j),r'Absolute error of velocity $v$ (m/s)')

    plotSolution(input_test[j,:,0], input_test[j,:,1], predictions[0,2,:].cpu().numpy(),'p_pred_test'+str(j),r'Prediction of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], output_test[j,:,2],'p_truth_test'+str(j),r'Ground truth of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0], input_test[j,:,1], np.abs(predictions[0,2,:].cpu().numpy()-output_test[j,:,2]),'p_abs_test'+str(j),r'Absolute error of gauge pressure (Pa)')

    rms_u += compute_rms_error(predictions.cpu().numpy(), 0, output_test[j,:,0])
    lrms_u += compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0])

    rms_v += compute_rms_error(predictions.cpu().numpy(), 1, output_test[j,:,1])
    lrms_v += compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1])

    rms_p += compute_rms_error(predictions.cpu().numpy(), 2, output_test[j,:,2])
    lrms_p += compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2])

    u_collection.append(compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0]))
    v_collection.append(compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1]))
    p_collection.append(compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2]))


print("Average RMS of Test for u: ", rms_u / len(test_idx))
print("Average Relative of Test for u: ", lrms_u / len(test_idx))
print()
print("Average RMS of Test for v: ", rms_v / len(test_idx))
print("Average Relative of Test for v: ", lrms_v / len(test_idx))
print()
print("Average RMS of Test for p: ", rms_p / len(test_idx))
print("Average Relative of Test for p: ", lrms_p / len(test_idx))
print()
print("Maximum relative error of test for u: ", max(u_collection))
print("Index: ",u_collection.index(max(u_collection)))
print()
print("Maximum relative error of test for v: ", max(v_collection))
print("Index: ",v_collection.index(max(v_collection)))
print()
print("Maximum relative error of test for p: ", max(p_collection))
print("Index: ",p_collection.index(max(p_collection)))
print()
print("Minimum relative error of test for u: ", min(u_collection))
print("Index: ",u_collection.index(min(u_collection)))
print()
print("Minimum relative error of test for v: ", min(v_collection))
print("Index: ",v_collection.index(min(v_collection)))
print()
print("Minimum relative error of test for p: ", min(p_collection))
print("Index: ",p_collection.index(min(p_collection)))

with open("u_collection.txt", "w") as file:
    for item in u_collection:
        file.write(f"{item}\n")

with open("v_collection.txt", "w") as file:
    for item in v_collection:
        file.write(f"{item}\n")

with open("p_collection.txt", "w") as file:
    for item in p_collection:
        file.write(f"{item}\n")
