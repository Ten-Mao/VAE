from torch import nn


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = nn.functional.relu(self.linear1(x))
        mu = self.linear2(hidden)
        log_var = self.linear3(hidden)
        return mu, log_var

