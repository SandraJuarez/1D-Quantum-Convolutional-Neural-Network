# Fourier Series Guided Design of Quantum Convolutional Neural Networks for Enhanced Time Series Forecasting

## Abstract
In this work, we apply 1D quantum convolution in the task of time series forecasting. By encoding multiple points into the quantum circuit to predict subsequent data, each point becomes a feature and the problem becomes a multidimensional one. Taking as basis previous theoretical works which demonstrated that Variational Quantum Circuits (VQCs) can be expressed as multidimensional Fourier series, the capabilities of different architectures and ansatz are explored. This analysis considers the concepts of circuit expressivity and the presence of barren plateaus. Analyzing the problem within the framework of the Fourier series enabled the incorporation of data reuploading in the architecture,
resulting in enhanced performance. Rather than a strict requirement for the number of free parameters to exceed the degrees of freedom of the Fourier series,
our findings suggest that even a limited number of parameters can produce Fourier functions of higher degrees. This highlights the remarkable expressive power of quantum circuits. This observation is also significant in reducing training
times. The ansatz with greater expressivity and number of non-zero Fourier coefficients consistently delivers favorable results across different scenarios, with performance metrics improving as the number of qubits increases.

## Files of the project
- load_data.py Scales data, creates the sequences and generates the training and testing tensors. Until now it only Works with the euro, mackey glass and legendre datasets. 
- quanvolution.py It contains the class QuanvLayer1D. Here we can find the circuit for the 1d quanvolution. The circuit can be set in the options 'no_reupload', 'parallel' and 'superparallel'. Also it contains 5 different ansatz for the trainable part. Details about each of the options can be found in the manuscript. Those options can be modified in the main file. If you want to modify the interface, diff_method or simulator this can be modified in quanvolution.py
- hybrid_circuit.py It integrates the quanvolutional layer with the classical part (activation function, max pooling and a linear layer to map to the predicted value
-training.py It contains the functions for performing training, tracking metrics with mlflow and saving .pt files. If you want to modify the number of epochs, mlflow parameters and folders names to sabe .pt files this can be done directly in this file. 
- main.ipynb Here you can train for the dataset, number of qubits, ansatz and architecture of your choice. 

## Secondary files
-barren_plateau.ipynb We can calculate the variance of the gradient
- kl_div.ipynb Calculates expressibility for the iddle circuit and for the circuits in the paper. 
- coefficients.ipynb Generates the plots for the accesible coefficients for the ansatz and architectures of the paper. 




## The complete work can be found in:
https://arxiv.org/abs/2404.15377