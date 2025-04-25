from hybrid_circuit_jax import CNN
#from Quanvolution_jax2 import QuanvLayer1D
from Quanvolution_jax2_multidim import QuanvLayer1D
import jax
import jax.numpy as jnp
import optax
import torch
import flax.linen as nn
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import mlflow
#from sklearn.metrics import mean_absolute_error
#import os
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
        sample_input = jnp.array(X_train[0])
        input_shape = sample_input.shape
        features=input_shape[0]
        net = QuanvLayer1D(kernel_size,n_layers,ansatz,out_size,out_channels,architecture,features)
        
        key2 = jax.random.PRNGKey(key)
        
        params = net.init(key2, jnp.ones((1,4,5)))
        #variables = net.init_with_output(key2, jnp.ones((2,2,3)))
        #f_jitted=jax.jit(nn.init(CNN,net))
        #variables = f_jitted(key2, jnp.ones(input_shape))
        #params = variables['params']
        #weights = jnp.ones([2,2,3])
        #params = {"weights": weights}
        opt = optax.adam(learning_rate=5e-4)
        opt_state = opt.init(params)
        for key, value in params.items():
            print(f"Param before {key}: {value}")
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
        @jax.jit
        def get_resta(inputs, targets,max_dataset,min_dataset):
            Predictions = net.apply(params,inputs)
            Predictions = Predictions*(max_dataset-min_dataset)+min_dataset
            targets=targets*(max_dataset-min_dataset)+min_dataset
            resta=Predictions-targets
            resta_l.append(resta)
            forecasted.append(Predictions)
            return resta_l,forecasted
        # Run the training loop
        epochs = 5 #for time complexity measurments set to 5
        Loss_hist=[]
        #for the rescaling
        #print('el shape del original es',original_dataset.shape)
        max_dataset=jnp.max(original_dataset,axis=1)
        #print('el max dataset',max_dataset)
        min_dataset=jnp.min(original_dataset,axis=1)
        #rescaled_target=targets*(max_dataset-min_dataset)+min_dataset
        # Record the start time
        #start_time = time.time()
        for epoch in range(epochs):
            if epoch==1:
              start_time = time.time()  
            print(f'Starting epoch {epoch+1}')
            epoch_loss = 0.0
            # Iterate over the training data
            resta_l=[]
            forecasted=[]
            for data in trainloader:
                inputs = data[0]
                targets = data[1]
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
                
                epoch_loss += inputs.shape[0] * loss
                if epoch==29:
                    
                    resta_l, forecasted=get_resta(inputs,targets,min_dataset,max_dataset)
                # Print statistics
            epoch_loss = epoch_loss/len(X_train)
            print('[Epoch %d] loss: %.6f' % (epoch + 1, epoch_loss))
            Loss_hist.append(epoch_loss)
        end_time = time.time()
        #Calculate and print the total execution time
        total_time = end_time - start_time 
        print('El tiempo de entrenamiento fue:',total_time)
        mlflow.log_metric('training time',total_time) 
        #uncomment when not measuring time complexity
        '''
        resta_l=np.array(resta_l)
        n_points_train=len(Y_train)
        RMSE = np.sqrt(np.sum(np.square(resta_l))/n_points_train)
        
        print('The RMSE is:',RMSE)
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
        resta_l=[]
        forecasted=[]
        for data in testloader:
            inputs = data[0]
            targets = data[1]
        
            resta_l,forecasted=get_resta(inputs,targets,min_dataset,max_dataset)
        resta_l=np.array(resta_l)
        forecasted=np.array(forecasted)
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/test_forecasted{run_Name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(forecasted, filenameLoss)
        print(np.max(original_dataset))
        RMSE = np.sqrt(np.sum(np.square(resta_l))/n_points_test)
        mlflow.log_metric('RMSE_TEST',RMSE)
        mae = (np.sum(np.abs(resta_l))/n_points_test)
        print("MAE=",mae)
        mlflow.log_metric('MAE_TEST',mae)
        MAPE = np.sum(np.absolute(resta_l)/np.absolute(targets))/n_points_test
        mlflow.log_metric('MAPE_TEST',MAPE)
        print("MAPE=",MAPE)
        '''
