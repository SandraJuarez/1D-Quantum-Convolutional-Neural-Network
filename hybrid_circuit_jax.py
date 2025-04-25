
import flax.linen as nn
from Quanvolution_jax import QuanvLayer1D
import numpy as np
import jax
#jax.config.update('jax_enable_x64', True)

np.random.seed(0)  


    
class CNN(nn.Module):
    kernel_size: int
    n_layers: int
    ansatz: str
    out_size: int
    out_channels: int
    architecture: str  

    @nn.compact
    def __call__(self, x):
        # Custom 1D Quantum Convolutional Layer
        quanv_layer = QuanvLayer1D(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            ansatz=self.ansatz,
            architecture=self.architecture
        )
        try:
            x = quanv_layer(x)
        except Exception as e:
            print(f"Error after QuanvLayer1D: {e}")
            raise e
        # Flatten the output
        x = x.reshape((x.shape[0], -1))
        # Fully connected layer
        x = nn.Dense(features=1)(x)       
        return x