from torch import nn


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, z):
        hidden = nn.functional.relu(self.linear1(z))
        reconstruction = nn.functional.sigmoid(self.linear2(hidden))
        return reconstruction