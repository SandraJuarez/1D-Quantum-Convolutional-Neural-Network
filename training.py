from hybrid_circuit import CNN
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
from sklearn.metrics import mean_absolute_error
import os
from pytorch_model_summary import summary
import time
import collections
torch.manual_seed(0)
np.random.seed(0)  




def train_testing_phase(X_train,Y_train,X_test,Y_test,kernel_size,n_layers,ansatz,out_size,out_channels,trainloader,run_Name,dataset,original_dataset,architecture):
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)
    #experiment_id = experiment.experiment_id
    with mlflow.start_run(run_name=run_Name):
        mlflow.log_param('kernel',kernel_size)
        mlflow.log_param('layers',n_layers)
        mlflow.log_param('ansatz',ansatz)
        mlflow.log_param('qubits',out_channels)
        net = CNN(kernel_size,n_layers,ansatz,out_size,out_channels,architecture)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
        sample_input = X_train[0]
        input_shape = sample_input.shape
        # Run the training loop
        epochs = 30
        Loss_hist=[]
        # Record the start time
        start_time = time.time()
        for epoch in range(epochs):
            net.train()
            #batch_size_summary = 1
            # show input shape
            #print(summary(net, torch.zeros((batch_size_summary, 4)), show_input=True))

            # show output shape
            #print(summary(net, torch.zeros((batch_size_summary, 4)), show_input=False))
            print(f'Starting epoch {epoch+1}')
            epoch_loss = 0.0
            # Iterate over the training data
            for data in trainloader:
                inputs = data[0].requires_grad_(False).type(torch.float)
                targets = data[1].requires_grad_(False).type(torch.float)
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = net(inputs)    
                # Compute loss
                loss = loss_function(outputs, targets.view(-1,1)) # The parameters of nn.MSELoss() must have the same size(shape).
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                epoch_loss += outputs.shape[0] * loss.item() # outputs.shape[0] = batch_size
                # loss.item() = loss calculated for each item in the batch, summed and then
                # divided by the size of the batch.
                # outputs.shape[0]*loss.item() is equal to the standard loss of the mini batch
                # (without the average)    

                # Print statistics
            epoch_loss = epoch_loss/len(X_train)
            print('[Epoch %d] loss: %.6f' % (epoch + 1, epoch_loss))
            Loss_hist.append(epoch_loss)

        # Process is complete.
        print('Training process has finished.')
        # Record the end time
        end_time = time.time()

        #Calculate and print the total execution time
        total_time = end_time - start_time 
        print(' El tiempo de entrenamiento es',total_time)
        mlflow.log_metric('training time',total_time) 
        Loss_hist_QCNN=np.array(Loss_hist)
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/Loss{run_Name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(Loss_hist_QCNN, filenameLoss)
        n_points_train = len(Y_train)
        forecasted = np.zeros((n_points_train))
        
        for k in range(n_points_train):
            forecasted[k] = net(X_train[k].type(torch.float))
       
        

        # Perform inverse transform
        actual_forecasted = forecasted*(np.max(original_dataset)-np.min(original_dataset))+np.min(original_dataset)
        Y_train=np.array(Y_train)
        actual_targets =  Y_train*(np.max(original_dataset)-np.min(original_dataset))+np.min(original_dataset)
       
        
        filenameTrain=f'{folder_name}/Train_forecasted{run_Name}.pt'
        torch.save(actual_forecasted, filenameTrain)
        RMSE = np.sqrt(np.sum(np.square(actual_forecasted-actual_targets))/n_points_train)
        mlflow.log_metric('RMSE_TRAIN',RMSE)
        print('In the training stage, the RMSE is:',RMSE)
        #####################################################
        ######################################################
        ##########TESTING ##############
        print('we begin testing')
        n_points_test = len(X_test)
        forecasted = np.zeros((n_points_test))
        for k in range(n_points_test):
            forecasted[k] = net(X_test[k].type(torch.float))
        actual_forecasted = forecasted*(np.max(original_dataset)-np.min(original_dataset))+np.min(original_dataset)
        Y_test=np.array(Y_test)
        actual_targets =   Y_test*(np.max(original_dataset)-np.min(original_dataset))+np.min(original_dataset)
        filenameTest=f'{folder_name}/Test_forecasted{run_Name}.pt'
        torch.save(actual_forecasted, filenameTest)
        
        
        
        RMSE = np.sqrt(np.sum(np.square(actual_forecasted-actual_targets))/n_points_test)
        print('RMSE=',RMSE)
        mlflow.log_metric('RMSE_TEST',RMSE)
        mae = mean_absolute_error(actual_forecasted,actual_targets)
        print("MAE=",mae)
        mlflow.log_metric('MAE_TEST',mae)
        MAPE = np.sum(np.absolute(actual_forecasted-actual_targets)/np.absolute(actual_targets))/n_points_test
        mlflow.log_metric('MAPE_TEST',MAPE)
        print("MAPE=",MAPE)
