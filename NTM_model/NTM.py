import torch
import torch.nn as nn
import torch.nn.functional as F

topic_num = 100
vec_size = 40535

device = 'cuda'

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        #encoder:
        self.fc1 = nn.Sequential(nn.Linear(vec_size, 768), nn.BatchNorm1d(768))
        self.fc2 = nn.Sequential(nn.Linear(768, 64), nn.BatchNorm1d(64))
        self.mean = nn.Linear(64, topic_num)
        self.var = nn.Linear(64, topic_num)

        #decoder:
        self.fc4 = nn.Linear(topic_num, 64)
        self.fc5 = nn.Sequential(nn.Linear(64, 768), nn.BatchNorm1d(768))
        self.fc6 = nn.Linear(768, vec_size)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.mean(x), self.var(x)

    def decoder(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        out = F.sigmoid(self.fc6(z))

        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        mean, log_var = self.encoder(inputs)

        z = self.reparameterize(mean, log_var)

        inputs_hat = self.decoder(z)

        return inputs_hat, mean, log_var, z

