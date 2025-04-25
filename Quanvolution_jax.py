
import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
from pennylane.templates import RandomLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers

#jax.config.update('jax_enable_x64', True)



class QuanvLayer1D(nn.Module):
    out_channels: int 
    kernel_size: int
    n_layers: int
    ansatz: str
    architecture: str
    sim_dev: str = "default.qubit"
    stride: int = 1
    seeds: int = 0
    padding: int = 1
    def setup(self):
        # Generate PRNG key
        #key = jax.random.PRNGKey(self.seeds)
        #key1, key2, key3 = jax.random.split(key, 3)
        
        # Initialize weights
        #if self.ansatz == 'strongly_layers':
            #weight_shapes = (self.n_layers + 1, self.out_channels, 3)
        #elif self.ansatz == 'nxn_params':
            #weight_shapes = (self.out_channels * self.out_channels, self.n_layers + 1, self.out_channels)
        #else:
            #weight_shapes = (self.n_layers + 1, self.out_channels, 3)
        
        # Define the weight parameters
        #self.params = self.param('weights', nn.initializers.xavier_uniform(), shape=weight_shapes)
        self.weights = self.param('weights', nn.initializers.normal(),
                                 (self.n_layers+1, self.out_channels,3))
        self.weights = jnp.asarray(self.weights, dtype=jnp.float32)
     

    
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
                
            
            # Measurement producing out_channels classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
        Circuit=circuit(inputs,weights)
        return Circuit
     
    
        
    @nn.compact
    def __call__(self, vector):
        try:
            bs, ch, l = vector.shape  # Ensure that vector is a JAX array
        except Exception as e:
            raise e  
        bs, ch, l = vector.shape # bs=batch_size, ch=channels, l=length
        if ch > 1:
            # We compute the mean along the channel dimension:
            vector = vector.mean(axis=-2).reshape(bs, 1, l)
        # We add padding:
        pad = ((0, 0), (0, 0), (self.padding, self.padding))
        vector = jnp.pad(vector, pad, mode='constant', constant_values=0)
        # l_out calculated from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html with dilation = 1              
        l_out = (l + 2*self.padding - self.kernel_size) // self.stride + 1
       
        out = jnp.ones((bs, self.out_channels, l_out))

        # Loop over the coordinates of the top pixel of kernel_size x 1 regions
        for b in range(bs):
            for k in range(0, l_out, self.stride):
                # Process a kernel_size x 1 region of the vector with a quantum circuit
                inputs=jnp.array(vector[b, 0, k:k + self.kernel_size])
                print('el shape de inputs es',inputs.shape)
                q_results = self.Qcircuit(inputs,self.weights)
                
                # Assign expectation values to different channels of the output pixel
                for c in range(self.out_channels):
                    out = out.at[b, c, k // self.stride].set(q_results[c])
        #print('el out',out)  
        out=nn.relu(out)     
        #we perform max pooling
        out=jnp.max(out, axis=-1, keepdims=True) 
        return out