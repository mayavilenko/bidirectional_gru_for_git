import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pandas as pd
import numpy as np
import statistics
import shutil

from model.GRU_model import *
from pipeline_config import *

def create_dataloader(category_train_df, category_test_df):
    x_train = category_train_df.iloc[:,:-1].to_numpy()
    y_train = category_train_df.iloc[:,-1].to_numpy()
    x_test = category_test_df.iloc[:,:-1].to_numpy()
    y_test= category_test_df.iloc[:,-1].to_numpy()

    x_train = torch.from_numpy(x_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)
    print('y test is: '+str(y_test))

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader =  DataLoader(train_dataset, batch_size=BatchSize, shuffle=False)
    test_dataloader =  DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    return train_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer):
    running_loss = 0
    model.train()
    predictions_list = []
    for inputs, labels in train_dataloader:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        #Changing input shape - last batch size can change so we define it as input.shape[0]
        inputs = inputs.view(inputs.shape[0], SequenceLength, Features) 
        #model prediction
        pred = model(inputs)
        #append batch predictions to predictions list
        predictions_list.append(pred.view(1,-1))
        # calculate loss
        loss = Criterion(pred, labels.view(-1,1))
        # calculate the gradient
        loss.backward()
        # update parameters
        optimizer.step()
        #Add to loss of batch to epoch train loss
        running_loss+=loss.item()
    # Calculte the epoch train loss
    epoch_train_loss = running_loss/len(train_dataloader.dataset)
    return epoch_train_loss


def evaluation_loop(model, test_dataloader):
     # Evaluation
    # Initiate test loss, accuracy and f1 score to zero
    test_loss = 0
    # Change model to eval mode
    model.eval()
    # we dont need to update weights, so we define no_grad() to save memory
    predictions_list = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            predictions_list.append(out.view(1,-1))
            test_batch_loss = Criterion(out, labels.view(-1,1))
            test_loss += test_batch_loss.item()
    # Calculate epoch loss
    epoch_predictions = torch.cat(predictions_list, dim=1)
    epoch_test_loss = test_loss/len(test_dataloader.dataset)
        
    return epoch_test_loss, epoch_predictions


def save_checkpoint(checkpoint, is_best, checkpoint_path, best_model_path):
    """
    checkpoint: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(checkpoint, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


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


def unify_model_weights(model):
    param_dict ={}
    for name, param in model.named_parameters():
        param_dict[name] = param

    param_dict['gru.bias_hh_l0'] = param_dict['gru.bias_hh_l0'].view(-1,1)
    param_dict['gru.bias_ih_l0'] = param_dict['gru.bias_ih_l0'].view(-1,1)
    unified_weights = torch.hstack((
            param_dict['gru.weight_ih_l0'],
            param_dict['gru.weight_hh_l0'],
            param_dict['gru.bias_ih_l0'],
            param_dict['gru.bias_hh_l0']))

    return unified_weights


def training_and_evaluation(model, train_dataloader, test_dataloader, optim, category, checkpoint_path, best_checkpoint_path):
   #results list
   train_loss_list = []
   test_loss_list = []

   #Create writer for using tesndorboard
   writer = SummaryWriter(log_dir=f'{TbDirectory}_{category}')

   min_test_loss = np.inf

   for epoch in range(Epochs):
      #initiate train epoch loss
      epoch_train_loss = training_loop(model, train_dataloader, optim)
      epoch_test_loss, epoch_test_predictions = evaluation_loop(model, test_dataloader)

      checkpoint = {
         'epoch': epoch + 1,
         'valid_loss_min': epoch_test_loss,
         'state_dict': model.state_dict(),
         'optimizer': optim.state_dict(),
        }
      
      # save checkpoint
      save_checkpoint(checkpoint, False, checkpoint_path, best_checkpoint_path)

      if epoch_test_loss <= min_test_loss:
         save_checkpoint(checkpoint, True, checkpoint_path, best_checkpoint_path)
         min_test_loss = epoch_test_loss

      train_loss_list.append(epoch_train_loss)
      test_loss_list.append(epoch_test_loss)

      # Display those measures on tensorboard
      writer.add_scalar(tag='loss/train', scalar_value=epoch_train_loss, global_step=epoch)
      writer.add_scalar(tag='loss/test', scalar_value=epoch_test_loss, global_step=epoch)
    
   results = {'train_loss': train_loss_list, 'test_loss': test_loss_list} 
   return results


def create_dict_of_best_model_per_category(categories_list, dir_path):
    basic_model = GRUModel(input_dim = Features, hidden_dim = HiddenSize, layer_dim = LayersDim, output_dim = OutputDim, dropout_prob = DropoutProb)
    basic_optimizer = torch.optim.AdamW(basic_model.parameters(), lr=Lr)
    basic_model.to(device)

    best_models_dict = {}

    for category in categories_list:
        ckp_path = dir_path+category+'.pt'
        model, optimizer, checkpoint, valid_loss_min = load_checkpoint(ckp_path, basic_model, basic_optimizer)
        best_models_dict[category] = model
        
    return best_models_dict


def get_best_predictions_for_each_category(best_models_dict, train_dataset_dict, test_dataset_dict):
    best_predictions_dict = {}

    for category in list(best_models_dict.keys()):
        model = best_models_dict[category]
        train_dataloader, test_dataloader = create_dataloader(train_dataset_dict[category], test_dataset_dict[category])
        _, epoch_predictions = evaluation_loop(model, test_dataloader)
        best_predictions_dict[category] = epoch_predictions

    return best_predictions_dict


def get_weights_per_category(category_id_list, category_id_to_name_dict, dir_path):
    basic_model = GRUModel(input_dim = Features, hidden_dim = HiddenSize, layer_dim = LayersDim, output_dim = OutputDim, dropout_prob = DropoutProb)
    basic_optimizer = torch.optim.AdamW(basic_model.parameters(), lr=Lr)
    basic_model.to(device)

    best_models_weights_dict = {}

    for category_id in category_id_list:
        category_name = category_id_to_name_dict[category_id]
        ckp_path = dir_path+category_name+'.pt'
        model, optimizer, checkpoint, valid_loss_min = load_checkpoint(ckp_path, basic_model, basic_optimizer)
        category_model_weights = unify_model_weights(model)
        best_models_weights_dict[category_id] = category_model_weights
        
    return best_models_weights_dict