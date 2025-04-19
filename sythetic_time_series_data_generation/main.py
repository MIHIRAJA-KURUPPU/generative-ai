import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

df =pd.read_csv('temperature_data.csv')
print(df.head())

temperature_data = df['Temperature']
print(type(temperature_data))

temperature_data = df['Temperature'].values
print(type(temperature_data))
print(temperature_data)
print(len(temperature_data))

np.random.seed(42)
seq_length = 30
batch_size = 32

class TimeSeriesDataset(Dataset):
    def __init__(self,data,seq_length):
        self.data = data
        self.seq_length=seq_length

    def __len__(self):
        return len(self.data)-self.seq_length

    def __getitem__(self,i):
        x = self.data[i:i+self.seq_length]
        y = self.data [i+self.seq_length]
        return torch.tensor(x,dtype=torch.float32).unsqueeze(-1),torch.tensor(y,dtype=torch.float32)

dataset = TimeSeriesDataset(temperature_data,seq_length)
dataloader =DataLoader(dataset, batch_size = batch_size, shuffle=False)
print(dataloader)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True) #(batch_size, seq_lenght, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        h_0 = torch.zeros(2,x.size(0),self.hidden_dim)    # x.size(0) gives batch_size
        c_0 = torch.zeros(2,x.size(0),self.hidden_dim)

        lstm_out,_ =self.lstm(x,(h_0,c_0))
        out = self.fc(lstm_out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True) #(batch_size, seq_lenght, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h_0 = torch.zeros(2,x.size(0),self.hidden_dim)
        c_0 = torch.zeros(2,x.size(0),self.hidden_dim)
        lstm_out,_ =self.lstm(x,(h_0,c_0))
        out = self.fc(lstm_out[:,-1,:])
        return out
#

#Parameters
input_dim = 1
output_dim = 1
hidden_dim = 64
num_epochs = 1500

#Initialize model
generator = Generator(input_dim, hidden_dim, output_dim, seq_length)
discriminator = Discriminator(input_dim, hidden_dim)

criterion = nn.BCEWithLogitsLoss()

OptimizerG = optim.Adam(generator.parameters(),lr=0.0001)
OptimizerD = optim.Adam(discriminator.parameters(), lr=0.0001)

# Training Loop

for epoch in range(num_epochs):
    g_loss_epoch =0
    d_loss_epoch =0


    for real_batch,_ in dataloader:
        batch_size = real_batch.size(0)
        # print(batch_size)
        real_labels = torch.ones(batch_size,1)  #(32,1)
        fake_labels =  torch.zeros(batch_size,1)

        # Train the discriminator
        OptimizerD.zero_grad()

        real_output = discriminator(real_batch)
        d_loss_real = criterion(real_output,real_labels)

        noise = torch.randn(batch_size, seq_length, input_dim)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real+ d_loss_fake
        d_loss.backward()
        OptimizerD.step()

        # Train Generator
        OptimizerG.zero_grad()
        noise = torch.randn(batch_size, seq_length, input_dim)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        OptimizerG.step()


        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()

    if epoch % 10 ==0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss_epoch/len(dataloader):.4f}, Loss G:  {g_loss_epoch/len(dataloader):.4f}")

num_sequences = len(temperature_data) - seq_length
noise = torch.randn(num_sequences, seq_length,input_dim)
generated_data = generator(noise).detach().numpy().flatten()[:num_sequences]
actual_data = temperature_data[seq_length: seq_length+num_sequences]  #[30:365]

# plot the actual and generated data

plt.figure(figsize=(12,6))
plt.plot(actual_data,label='Actual Data', color='blue')
plt.plot(range(len(actual_data)),generated_data, label='Generated Data', color='orange')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.legend()
plt.show()

mse = mean_squared_error(actual_data,generated_data)
mae = mean_absolute_error(actual_data, generated_data)
correlation = np.corrcoef(actual_data, generated_data)[0,1]

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Correllation Coeff: {correlation:.4f}")