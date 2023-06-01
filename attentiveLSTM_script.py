# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
#%cd drive/MyDrive/energy_project

# %%
device = "mps"

# %%
from models.attention_exp_LSTM.dataset import energyProject_dataset
from models.attention_exp_LSTM.network import attentiveLSTM_model

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import yaml
import wandb
from typing import Dict

# %%
def nested_dict(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict


#load dataset object file
with (open('data/dataset_class.pkl', "rb")) as openfile:
    dataset = pickle.load(openfile)

dataset.activitivity_encoding_mode = 2 #or any value
# Splitting the dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=100, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)

# %%
#setup wandb stuff
with open('models/attention_exp_LSTM/config.yaml', 'r') as stream:
    try:
        sweep_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# %%
y_idx=0

# %%
scaler = dataset.column_scalers[dataset.target_labels[y_idx]]

# %%
def train(config: Dict = None):
    with wandb.init(config=config):
        config = wandb.config
        config = nested_dict(config)
        optimizer_config = config["optimizer"]
        model = attentiveLSTM_model(espai_emb_dim=config["espai_emb_dim"],hidden_dim=config['hidden_size'],lstm_nl=config['num_layers'],nheads=config['heads'],attnFCdim=config['attnFCdim'])
        model.init_weights()
        model.to(device)

        criterion =  nn.MSELoss()

        if optimizer_config["type"] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'])

        num_epochs = 80
        best_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            training_losses = [] # renamed from epoch_losses
            progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch,data in progress_bar:
                ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
                y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now

                optimizer.zero_grad()
                
                #current batch size size
                b_sz = ocu_emb.shape[0]

                #note the dataloader with a batch of 100 when reachs the end expects a batch of 60
                h, c = model.init_hidden(b_sz) # Start with a new state in each batch            
                h = h.to(device)
                c = c.to(device)
                y_pred, h,c= model(ocu_emb, espai_enc, general_data, h, c)
                y_pred = y_pred[:,0,0]

                loss = criterion(y_pred,y)  #cross entropy loss needs (N,C,seq_lenght)
                loss.backward()
                optimizer.step()    

                training_losses.append(loss.item())
                progress_bar.set_postfix({'Batch Loss': loss.item()})

            average_training_loss = sum(training_losses) / len(training_losses) # renamed from avg_loss
            average_training_loss = np.power(dataset.denormalize_values(np.sqrt(average_training_loss),scaler),2)
            wandb.log({'Train_Epoch_Loss': average_training_loss})

            model.eval()  
            with torch.no_grad():  
                validation_losses = [] # renamed from val_losses
                for batch in tqdm(val_dataloader, desc='Validation'):
                    ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
                    y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now

                    #current batch size size
                    b_sz = ocu_emb.shape[0]

                    #note the dataloader with a batch of 100 when reachs the end expects a batch of 60
                    h, c = model.init_hidden(b_sz) # Start with a new state in each batch            
                    h = h.to(device)
                    c = c.to(device)
                    y_pred, h,c= model(ocu_emb, espai_enc, general_data, h, c)
                    y_pred = y_pred[:,0,0]
                    
                    loss = criterion(y_pred,y) 
                    validation_losses.append(loss.item())

                average_validation_loss = sum(validation_losses) / len(validation_losses) # renamed from avg_val_loss
                average_validation_loss = np.power(dataset.denormalize_values(np.sqrt(average_validation_loss),scaler),2)
                wandb.log({'Validation_Epoch_Loss': average_validation_loss})

            if average_training_loss < best_loss:
                best_loss = average_training_loss
                torch.save(model.state_dict(), 'models/attention_exp_LSTM/tranformLSTM.pt')
                wandb.save('gru_model.pt')
                print(f"Model saved at {'gru_model.pt'}")

        wandb.finish()

# %%
#!wandb login

# %%
sweep_id = wandb.sweep(sweep_config, project="energy_project_uab")
sweep_id

# %%
wandb.agent(sweep_id, function=train)


