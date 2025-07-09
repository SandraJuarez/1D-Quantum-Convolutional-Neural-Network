from hybrid_circuit_jax import CNN
from Quanvolution_jax2_multidim_vectorized import QuanvLayer1D
#from Quanvolution_jax2 import QuanvLayer1D
import jax
import jax.numpy as jnp
import optax
import torch
import matplotlib.pyplot as plt
import flax.linen as nn
import numpy as np
import mlflow
import os
import time
from cnn import ClassicalCNN as CNN

jax.config.update('jax_enable_x64', True)


def train_testing_phase(X_train, Y_train, X_test, Y_test,
                        kernel_size, n_layers, ansatz, out_size, out_channels,
                        trainloader, testloader, run_Name, dataset,
                        original_dataset, architecture, key, init, model,convergence=False):
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(run_name=run_Name):
        ### Log hyperparameters
        mlflow.log_params({
            'kernel': kernel_size,
            'layers': n_layers,
            'ansatz': ansatz,
            'qubits': out_channels,
            'init': init,
            'model': model
        })

        ### Build model
        sample_input = jnp.array(X_train[0]).reshape((1,) + X_train[0].shape)
        if model == 'QCNN':
            net = QuanvLayer1D(kernel_size, n_layers, ansatz, out_size, out_channels, architecture) #features  
        elif model == 'CNN':
            net = CNN(kernel_size, features=sample_input.shape[0])

        key2 = jax.random.PRNGKey(key)
        params = net.init(key2, jnp.ones(sample_input.shape))
        opt = optax.adam(learning_rate=5e-4)
        opt_state = opt.init(params)

        ### Precompute target normalization (VERY IMPORTANT)
        max_target = jnp.max(Y_train)
        min_target = jnp.min(Y_train)

        ### Training step
        @jax.jit
        def train_step(params, opt_state, inputs, targets):
            def mse_loss(params, inputs, targets):
                predictions = net.apply(params, inputs)
                loss = jnp.mean((predictions - targets) ** 2)
                return loss
            loss, grads = jax.value_and_grad(mse_loss)(params, inputs, targets)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        ### Validation step
        @jax.jit
        def validation_step(params, inputs, targets):
            predictions = net.apply(params, inputs)
            loss = jnp.mean((predictions - targets) ** 2)
            return loss

        ### Denormalization function
        def denormalize(predictions, min_val, max_val):
            return predictions * (max_val - min_val) + min_val

        ### Training loop
        epochs = 30
        variance_threshold = 5e-6
        loss_history = []
        val_loss_history = []
        previous_losses = []
        converged = False

        print(f"Starting training...")

        for epoch in range(epochs):
            if epoch == 0:
                start_time = time.time()

            print(f'\nEpoch {epoch+1}/{epochs}')
            train_loss = 0.0
            train_batch_losses = []

            for data in trainloader:
                inputs, targets = data
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
                train_loss += loss * inputs.shape[0]
                train_batch_losses.append(loss)

            train_loss /= len(X_train)
            loss_history.append(train_loss)

            ### Validation
            val_loss = 0.0
            for val_data in testloader:
                val_inputs, val_targets = val_data
                val_loss += validation_step(params, val_inputs, val_targets) * val_inputs.shape[0]
            val_loss /= len(X_test)
            val_loss_history.append(val_loss)

            previous_losses.append(val_loss)
            if convergence:
                ### Convergence check
                if epoch >= 10 and len(previous_losses) >= 5:
                    loss_var = np.var(previous_losses[-5:])
                    print(f"Validation Loss Variance (last 5 epochs): {loss_var:.8e}")
                    if loss_var < variance_threshold:
                        print(f"Convergence achieved at epoch {epoch+1}")
                        convergence_ep = epoch + 1
                        converged = True
                        break

            print(f"Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")

        if not converged:
            print("Training completed without reaching variance threshold.")
            convergence_ep = epochs

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        ### Save losses
        folder_name = dataset
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.save(np.array(loss_history), f'{folder_name}/LossT_{run_Name}.pt')
        torch.save(np.array(val_loss_history), f'{folder_name}/LossV_{run_Name}.pt')

        ### Save metrics
        mlflow.log_metric('training time', total_time)
        mlflow.log_metric('final_train_loss', train_loss)
        if convergence:
            mlflow.log_metric('convergence_epoch', convergence_ep)

        ### Testing
        print("\nStarting Testing...")

        preds_norm = net.apply(params, X_test)
        preds = denormalize(preds_norm, min_target, max_target)
        true_vals = denormalize(Y_test, min_target, max_target)

        residuals = preds - true_vals

        RMSE = np.sqrt(jnp.mean(residuals**2))
        MAE = jnp.mean(jnp.abs(residuals))
        preds_np = np.array(preds).flatten()
        true_vals_np = np.array(true_vals).flatten()
        epsilon = 1e-8
        valid = np.abs(true_vals_np) > epsilon
        MAPE = np.mean(np.abs(preds_np[valid] - true_vals_np[valid]) / np.abs(true_vals_np[valid]))


        print(f"Test RMSE: {RMSE:.6f}")
        print(f"Test MAE: {MAE:.6f}")
        print(f"Test MAPE: {MAPE:.6f}")

        mlflow.log_metrics({
            'RMSE_test': float(RMSE),
            'MAE_test': float(MAE),
            'MAPE_test': float(MAPE)
        })

        ### Save predictions
        torch.save(np.array(preds), f'{folder_name}/test_forecasted_{run_Name}.pt')

        print("Training and Testing phase completed successfully!")

        # Ensure both are numpy arrays
        preds_np = np.array(preds).flatten()
        true_vals_np = np.array(true_vals).flatten()

        plt.figure(figsize=(8, 5))
        plt.plot(true_vals_np, label='Real', linewidth=2)
        plt.plot(preds_np, label='Predicted', linewidth=2)
        plt.legend()
        plt.title("Predicted vs Real Values")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plot_path = f'{folder_name}/plot_{run_Name}.png'
        plt.savefig(plot_path)
        plt.show()
