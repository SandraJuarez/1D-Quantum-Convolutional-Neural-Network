
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import collections
torch.manual_seed(0)
np.random.seed(0)  


def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples = 1):
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild 
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t 
    # Initial conditions for the history of the system
    timeseries = 1.2
    
    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))
        
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        
        # Squash timeseries through tanh
        # inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples

# class to construct a custom dataloader: 
class CustomDataset(Dataset):
    def __init__(self, inputs, labels=None, transforms=None):
        self.X = inputs
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X[i, :]
        
        # Add an extra dimension to the data
        data = data[np.newaxis, :]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        
def data(dataset):
    scaler = MinMaxScaler()
    if dataset=='euro':
        data = np.genfromtxt('USD-EURO-Time-Series.csv', delimiter=',')
        # The data is normalized is scaled in the range [0,1]
        scaled_data_usd_euro = (data-np.min(data))/(np.max(data)-np.min(data))#scaler.fit_transform(data.reshape(-1,1))

        scaled_data = np.zeros((376,6))
        for t in range(4,380):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_usd_euro[t-4],scaled_data_usd_euro[t-3],scaled_data_usd_euro[t-2],
                                    scaled_data_usd_euro[t-1],scaled_data_usd_euro[t],scaled_data_usd_euro[t+1]]
            # We use the first 300 points to train the model and the rest 
            # for the testing phase:
        scaled_data_train = scaled_data[0:300,0:6]
        scaled_data_test = scaled_data[300:,0:6]
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]  
    elif dataset=='mackey':
        data = mackey_glass(sample_len=1024,seed=0)[0]
        scaled_data_mackey_glass = (data-np.min(data))/(np.max(data)-np.min(data)) #scaler.fit_transform(data)
        scaled_data = np.zeros((1000,5))
        for t in range(18,1018):
            idx = t-18 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_mackey_glass[t-18], scaled_data_mackey_glass[t-12],
                                    scaled_data_mackey_glass[t-6], scaled_data_mackey_glass[t],
                                    scaled_data_mackey_glass[t+6]]
        # We use the first 500 points to train the model and the rest 
        # for the testing phase:
        scaled_data_train = scaled_data[0:750,0:5]
        scaled_data_test = scaled_data[750:,0:5]
        X_train = scaled_data_train[:,:4]  
        Y_train = scaled_data_train[:,4]
        X_test = scaled_data_test[:,:4]  
        Y_test = scaled_data_test[:,4]
    elif dataset=='legendre3':
        np.random.seed(42)
        x_values = np.linspace(0, 100, 1000)
        # Calculate the third Legendre polynomial without noise
        legendre_poly = 1/8 * (3 * np.cos(x_values) - 5*np.cos(3*x_values))
        # Add seeded random noise to the polynomial
        noise = np.random.normal(0, 0.1, size=len(x_values))
        legendre_poly_with_noise = legendre_poly + noise
        # Store the points in a NumPy array
        data = np.squeeze(np.column_stack((legendre_poly_with_noise)))
        scaled_data_legendre = (data-np.min(data))/(np.max(data)-np.min(data))#scaler.fit_transform(data.reshape(-1,1))
        scaled_data = np.zeros((996,6))
        for t in range(4,999):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_legendre[t-4],scaled_data_legendre[t-3],scaled_data_legendre[t-2],
                                    scaled_data_legendre[t-1],scaled_data_legendre[t],scaled_data_legendre[t+1]]
        # We use the first 300 points to train the model and the rest 
        # for the testing phase:
        scaled_data_train = scaled_data[0:750,0:6]
        scaled_data_test = scaled_data[750:,0:6]
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]


    # We convert to torch tensor:
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)

    train_data = CustomDataset(X_train, Y_train)
    test_data = CustomDataset(X_test, Y_test)
    # Dataloaders
    batch_size_train = 1
    batch_size_test = 1
    trainloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    return X_train,Y_train,X_test,Y_test,trainloader,testloader,data