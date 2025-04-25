
import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
from pennylane.templates import RandomLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers

jax.config.update('jax_enable_x64', True)


class QuanvLayer1D(nn.Module):
    kernel_size: int
    n_layers: int
    ansatz: str
    out_size: int
    out_channels: int
    architecture: str  
    features:int
    sim_dev: str = "default.qubit"
    stride: int = 1
    seeds: int = 0
    padding: int = 1
    
    def setup(self):
        
        seed=4
        key = jax.random.PRNGKey(seed)

        
        # Initialize weights
        if self.ansatz == 'strongly_layers':
            self.weights = self.param('weights', nn.initializers.normal(),
                                 (self.n_layers+1, self.out_channels,3))
        elif self.ansatz == 'nxn_params':
            self.weights = self.param('weights', nn.initializers.normal(),
                                 (self.out_channels * self.out_channels, self.n_layers + 1, self.out_channels))
        else:
            self.weights = self.param('weights', nn.initializers.normal(),
                                 (self.n_layers+1, self.out_channels))
        
        self.weights=jnp.mod(self.weights,2*jnp.pi)
        #self.weights = jnp.asarray(self.weights, dtype=jnp.float32)
     

    
    # random circuits
    def Qcircuit(self, inputs,weights):
        device = qml.device('default.qubit', wires=self.out_channels)
        #weights = self.weights
        @qml.qnode(device=device) #  device= "default.qubit" or "lightning.qubit" ##* interface='jax-jit'
        def circuit(inputs, weights):
            def custom_ansatz(i,weights):
                for j in range(self.out_channels):
                    qml.RX(weights[i,j], wires=j)
                    if j!=self.out_channels-1:
                        seq=[j,j+1]
                    else:
                        seq=[0,j]
                    qml.CNOT(wires=seq)
                    qml.RY(weights[i,j], wires=j)
                    qml.CNOT(wires=seq)
                    qml.RZ(weights[i,j], wires=j)

            def nxn_params(self,i,weights):
                nxn=int(self.out_channels*self.out_channels)
                for j in range(self.out_channels):
                    for n in range(nxn):
                        qml.RX(weights[n,i,j], wires=j)
                    if j!=self.out_channels-1:
                        seq=[j,j+1]
                    else:
                        seq=[0,j]
                    qml.CNOT(wires=seq)
                    for n in range(nxn):
                        qml.RY(weights[n,i,j], wires=j)
                    qml.CNOT(wires=seq)
                    for n in range(nxn):
                        qml.RZ(weights[n,i,j], wires=j)
            wires_list=list(range(self.out_channels))
            ## LAYER 0 ####
            if self.ansatz=='custom_layers':
                custom_ansatz(0,weights)

            elif self.ansatz=='nxn_params':
                nxn_params(0,weights)

            elif self.ansatz=='random_layers':
                pesos=weights[0,:]
                pesos=pesos.reshape(1,-1)
                RandomLayers(pesos,wires=wires_list,seed=1234)
            elif self.ansatz=='basic_layers':
                pesos=weights[0,:]
                pesos=pesos.reshape(1,-1)
                BasicEntanglerLayers(pesos,wires=wires_list)
            elif self.ansatz=='strongly_layers':
                pesos=weights[0,:]
                pesos=pesos.reshape(1,self.out_channels,-1)
                StronglyEntanglingLayers(pesos,wires=wires_list)
            
            ### ENCODING #####################################
            if self.architecture=='no_reupload':
                w=0
                for inp in range(self.kernel_size):
                    wi=int(wires_list[w+inp])
                    qml.RY(inputs[inp],wires=wi)

            for i in range(1,self.n_layers+1):
                w=0
                if self.architecture=='super_parallel' or self.architecture=='parallel':
                    for b in range(self.out_channels//self.kernel_size): #the blocks in which the input repeats "vertically"
                        for inp in range(self.kernel_size):
                            wi=int(wires_list[w+inp])
                            qml.RY(inputs[inp],wires=wi)
                        w=w+self.kernel_size
                
                

                
                ############################################################
                ##### TRAINABLE PART ##################################
                
                if self.ansatz=='custom_layers':
                    custom_ansatz(i,weights)
                elif self.ansatz=='nxn_params':
                    nxn_params(i,weights)

                elif self.ansatz=='random_layers':
                    pesos=weights[i,:]
                    pesos=pesos.reshape(1,-1)
                    RandomLayers(pesos,wires=wires_list,seed=1234)
                elif self.ansatz=='basic_layers':
                    pesos=weights[i,:]
                    pesos=pesos.reshape(1,-1)
                    BasicEntanglerLayers(pesos,wires=wires_list)
                elif self.ansatz=='strongly_layers':
                    pesos=weights[i,:]
                    pesos=pesos.reshape(1,self.out_channels,-1)
                    StronglyEntanglingLayers(pesos,wires=wires_list)
            
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
        
        bs, ch, l = inputs.shape 
        circuit = jax.jit(circuit)    
        #if ch > 1:
        # We compute the mean along the channel dimension:
            #inputs = inputs.mean(axis=-2).reshape(bs, 1, l)
            # We add padding:
        pad = ((0, 0), (0, 0), (self.padding, self.padding))
        inputs = jnp.pad(inputs, pad, mode='constant', constant_values=0)
        # Measurement producing out_channels classical output values
        l_out = (l + 2*self.padding - self.kernel_size) // self.stride + 1   
        out = jnp.ones((bs, self.out_channels, l_out))
        for b in range(bs):
            
            for k in range(0, l_out, self.stride):
                # Process a kernel_size x 1 region of the vector with a quantum circuit
                q_results=0
                for c in range(ch):  #loop in number of features
                    inp=jnp.array(inputs[b, c, k:k + self.kernel_size])
                    q_results = jnp.asarray(circuit(inp,weights))+q_results
                    
                
                # Assign expectation values to different channels of the output pixel
                for q in range(self.out_channels):
                    out = out.at[b, q, k // self.stride].set(q_results[q])
        return out
     
    
        
    @nn.compact
    def __call__(self, input):
        
        out=self.Qcircuit(input,self.weights)
        out=nn.relu(out)     
        #we perform max pooling
        out=jnp.max(out, axis=-1, keepdims=True) 
        #print('el shape despues de maxpool',out.shape)
        # Flatten the output
        out = out.reshape((out.shape[0], -1))
        #print('el shape despues de reshape',out.shape)
        # Fully connected layer
        #features=out.shape[-1]
        out = nn.Dense(self.features)(out)    #features=1 because we are predicting only 1 point
        print('el shape despues de dense',out.shape)
        return out