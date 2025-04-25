import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import torch.nn.functional as F
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers
import importlib
torch.manual_seed(0)
np.random.seed(0)           # Seed for NumPy random number generator


class QuanvLayer1D(nn.Module):
    def __init__(self, out_channels, kernel_size,  n_layers ,ansatz,architecture,sim_dev="default.qubit", stride=1,seed=1234,padding=1):
        super(QuanvLayer1D, self).__init__()       
        # init device
        self.wires = out_channels # We use n qubits to obtain n out_channels
        self.dev = qml.device(sim_dev, wires=self.wires)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.n_layers=n_layers
        self.ansatz=ansatz
        
        if seed is None:
            seed = np.random.randint(low=0, high=10e6)
            

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

        def nxn_params(i,weights):
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
        # random circuits
        @qml.qnode(device=self.dev, interface="torch", diff_method="backprop") #  device= "default.qubit" or "lightning.qubit"
        def circuit(inputs, weights):
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
            if architecture=='no_reupload':
                w=0
                for inp in range(self.kernel_size):
                    wi=int(wires_list[w+inp])
                    qml.RY(inputs[inp],wires=wi)

            for i in range(1,self.n_layers+1):
                w=0
                if architecture=='super_parallel' or architecture=='parallel':
                    for b in range(self.out_channels//self.kernel_size): #the blocks in which the input repeats "vertically"
                        for inp in range(self.kernel_size):
                            wi=int(wires_list[w+inp])
                            qml.RY(inputs[inp],wires=wi)
                        w=w+kernel_size
                
                

               
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

        if self.ansatz=='strongly_layers':
            weight_shapes={"weights": [n_layers+1, out_channels,3]}
        elif self.ansatz=='nxn_params':
            weight_shapes={"weights": [out_channels*out_channels,n_layers+1, out_channels]}
        else:
            weight_shapes = {"weights": [n_layers+1, out_channels]} 
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        
    
    def forward(self, vector):
        bs, ch, l = vector.size() # bs=batch_size, ch=channels, l=length

        if ch > 1:
            # We compute the mean along the channel dimension:
            vector = vector.mean(axis=-2).reshape(bs, 1, l)
        # We add padding:
        pad = (self.padding, self.padding, 0, 0, 0, 0)
        vector = F.pad(vector, pad, "constant", value = 0)
        # l_out calculated from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html with dilation = 1              
        l_out = (l + 2*self.padding - self.kernel_size) // self.stride + 1
       
        out = torch.zeros((bs, self.out_channels, l_out))

        # Loop over the coordinates of the top pixel of kernel_size x 1 regions
        for b in range(bs):
            for k in range(0, l_out, self.stride):
                # Process a kernel_size x 1 region of the vector with a quantum circuit
                q_results = self.circuit(
                    inputs=torch.Tensor(vector[b, 0, k:k + self.kernel_size])
                )
                # Assign expectation values to different channels of the output pixel
                for c in range(self.out_channels):
                    out[b, c, k // self.stride] = q_results[c]     
                
        return out