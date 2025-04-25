from hybrid_circuit_jax import CNN
#from Quanvolution_jax2 import QuanvLayer1D
#from Quanvolution_jax2_multidim import QuanvLayer1D
from Quanvolution_jax2_multidim_vectorized import QuanvLayer1D
import jax
import jax.numpy as jnp
import optax
import torch
import flax.linen as nn
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import mlflow
#from sklearn.metrics import mean_absolute_error
import os
import time
from jax import random

jax.config.update('jax_enable_x64', True)
def train_testing_phase(X_train,Y_train,X_test,Y_test,kernel_size,n_layers,ansatz,out_size,out_channels,trainloader,testloader,run_Name,dataset,original_dataset,architecture,key):
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)
    #experiment_id = experiment.experiment_id
    with mlflow.start_run(run_name=run_Name):
        mlflow.log_param('kernel',kernel_size)
        mlflow.log_param('layers',n_layers)
        mlflow.log_param('ansatz',ansatz)
        mlflow.log_param('qubits',out_channels)
        #net = CNN(kernel_size,n_layers,ansatz,out_size,out_channels,architecture)
        sample_output = jnp.array(Y_train[0])
        output_shape = sample_output.shape
        sample_input = jnp.array(X_train[0,:,:])
        input_shape = sample_input.shape
        input_shape = (1,) + input_shape
        print(input_shape)
        features=input_shape[0]
        net = QuanvLayer1D(kernel_size,n_layers,ansatz,out_size,out_channels,architecture,features)
        
        key2 = jax.random.PRNGKey(key)
        print('el shape del inp',input_shape)
        params = net.init(key2, jnp.ones(input_shape))
        #variables = net.init_with_output(key2, jnp.ones((2,2,3)))
        #f_jitted=jax.jit(nn.init(CNN,net))
        #variables = f_jitted(key2, jnp.ones(input_shape))
        #params = variables['params']
        #weights = jnp.ones([2,2,3])
        #params = {"weights": weights}
        opt = optax.adam(learning_rate=5e-4)
        opt_state = opt.init(params)
        #for key, value in params.items():
            #print(f"Param before {key}: {value}")
        @jax.jit
        def train_step(params, opt_state, inputs, targets):
            def mse_loss(params,inputs, targets):
                predictions = net.apply(params,inputs)
                loss = jnp.mean((predictions - targets) ** 2)
                return loss
            loss, grads = jax.value_and_grad(mse_loss)(params,inputs,targets)
            #print(f"Gradients for QuanvLayer1D: {grads['QuanvLayer1D']}")
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            #print("Gradients for QuanvLayer1D:", grads)
            return params, opt_state, loss
        resta_l=[]
        @jax.jit
        def validation_step(params, inputs, targets):
            """Calculate the validation loss for a batch."""
            predictions = net.apply(params, inputs)
            loss = jnp.mean((predictions - targets) ** 2)  # MSE Loss
            return loss
        @jax.jit
        def get_resta(inputs, targets,max_dataset,min_dataset):
            Predictions = net.apply(params,inputs)
            print('esta es el predictions',Predictions)
            Predictions = Predictions*(max_dataset-min_dataset)+min_dataset
            targets=targets*(max_dataset-min_dataset)+min_dataset
            resta=Predictions-targets
            resta_l.append(resta)
            forecasted.append(Predictions)
            return resta_l,forecasted
        # Run the training loop
        epochs = 200 #for time complexity measurments set to 5
        Loss_hist=[]
        #for the rescaling
        #print('el shape del original es',original_dataset.shape)
        max_dataset=jnp.max(original_dataset,axis=0)
        #print('el max dataset',max_dataset)
        min_dataset=jnp.min(original_dataset,axis=0)
        #print('el max dataset y el min dataset son',max_dataset,min_dataset)
        #rescaled_target=targets*(max_dataset-min_dataset)+min_dataset
        # Record the start time
        #start_time = time.time()
        loss_history = []
        val_loss_history = []
        variance_threshold = 5e-6  # Example threshold for loss variance
        converged = False
        convergence_ep = -1
        
        for epoch in range(epochs):
            if epoch == 0:
                start_time = time.time()

            print(f'\nStarting epoch {epoch + 1}')
            epoch_loss = 0.0
            epoch_losses = []  # To store loss for each batch in the epoch
            
            # Training Loop
            for data in trainloader:
                inputs = data[0]
                targets = data[1]
                
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
                epoch_loss += inputs.shape[0] * loss
                epoch_losses.append(loss)  # Store batch loss
            
            # Compute average training loss for the epoch
            epoch_loss = epoch_loss / len(X_train)
            loss_history.append(epoch_loss)  # Store epoch loss history
            
           
            
            # Validation Loss Calculation
            val_loss = 0.0
            for val_data in testloader:
                val_inputs = val_data[0]
                val_targets = val_data[1]
                val_loss += validation_step(params, val_inputs, val_targets) * val_inputs.shape[0]
                epoch_losses_val.append(val_loss)
            val_loss /= len(X_test)  # Average validation loss
            val_loss_history.append(val_loss)
            #variance in the batch of current epoch
            loss_variance = np.var(epoch_losses_val)
            
            # Print epoch statistics
            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.6f}, Variance = {loss_variance:.6e}, Validation Loss = {val_loss:.6f}")
            
            # Convergence Check Based on Loss Variance
            if loss_variance < variance_threshold:
                print(f"Convergence reached at epoch {epoch + 1}. Stopping training.")
                converged = True
                convergence_ep = epoch
                break  # Exit training loop

        # Final Check if Convergence Was Not Reached
        if not converged:
            print("\nTraining completed without reaching the variance threshold.")
        end_time = time.time()
        #Calculate and print the total execution time
        total_time = end_time - start_time 
        print('El tiempo de entrenamiento fue:',total_time)
        mlflow.log_metric('number of epochs',convergence_ep)
        mlflow.log_metric('training time',total_time)
        mlflow.log_metric('loss',epoch_loss) 
        mlflow.log_metric('convergence epoch',convergence_ep) 
        mlflow.log_metric('train loss vector',loss_history)
        mlflow.log_metric('val loss vector',val_loss_history)
        

        
        # Process is complete.
        print('Training process has finished.')
        for key, value in params.items():
            print(f"Param after {key}: {value}")
        Loss_hist_QCNN=np.array(Loss_hist)
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/Loss{run_Name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(Loss_hist_QCNN, filenameLoss)
        n_points_train = len(Y_train)
        print('Los puntos de entrenamiento son',n_points_train)
        
        #####################################################
        ######################################################
        ##########TESTING ##############
        n_points_test = len(Y_test)
        print('we begin testing')
        print('el shape de xtest',X_test.shape)
        print('el shape de ytest',Y_test.shape)
        resta_l=[]
        forecasted=[]
        #for data in testloader:
            #inputs = data[0]
            #targets = data[1]
        print(X_test.shape)
        resta_l,forecasted=get_resta(X_test,Y_test,min_dataset,max_dataset)
        resta_l=np.array(resta_l)
        #print(resta_l)
        forecasted=np.array(forecasted)
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/test_forecasted{run_Name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(forecasted, filenameLoss)
        #print(np.max(original_dataset))
        RMSE = np.sqrt(np.sum(np.square(resta_l))/n_points_test)
        print('RMSE',RMSE)
        mlflow.log_metric('RMSE_TEST',RMSE)
        mae = (np.sum(np.abs(resta_l))/n_points_test)
        print("MAE=",mae)
        mlflow.log_metric('MAE_TEST',mae)
        #print('los ytest',np.absolute(Y_test))
        #print('los points',n_points_test)
        MAPE = np.sum(np.absolute(resta_l)/np.absolute(Y_test))/n_points_test
        mlflow.log_metric('MAPE_TEST',MAPE)
        print("MAPE=",MAPE)
        
        