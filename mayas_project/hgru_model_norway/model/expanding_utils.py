import torch
import numpy as np
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from model.config import *

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

def create_dataloader(horizon_test_df, horizon):
    columns_pt1 =  ['Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t']
    columns_pt2 = []
    for h in range(horizon+1):
        columns_pt2.append('Inflation t+'+str(1+h))
        
    columns = columns_pt1[horizon:] + columns_pt2
    print(f'the number of columns is: {len(columns)}')
    print(f'the columns: {columns}')
    
    horizon_test_df = horizon_test_df[columns]
    
    x = horizon_test_df.iloc[:,:-1].to_numpy()
    y = horizon_test_df.iloc[:,-1].to_numpy()

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    
    dataset = TensorDataset(x, y)
    dataloader =  DataLoader(dataset, batch_size=BatchSize, shuffle=False)
    print(f'x shape is: {x.shape}')
    print(f'y shape is: {y.shape}')
    return dataloader


def save_checkpoint(checkpoint, path):
    """
    checkpoint: checkpoint we want to save
    path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(checkpoint, path)
    

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def get_predictions_on_test_set(model, dataloader):
    # Evaluation
    # Change model to eval mode
    model.eval()
    predictions_list = []
    # we dont need to update weights, so we define no_grad() to save memory
    with torch.no_grad():
        for inputs, labels in dataloader:
            print(f'input shape is: {inputs.shape}')
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(Device), labels.to(Device)
            out = model(inputs)
            predictions_list.append(out.view(1,-1))
    # Calculate epoch loss
    epoch_predictions = torch.cat(predictions_list, dim=1)
    return epoch_predictions



    
    
