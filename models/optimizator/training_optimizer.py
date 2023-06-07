# %%
from icecream import ic

# %%
device = "cuda"

# %%
from models.attention_exp_LSTM.dataset import energyProject_dataset
from models.attention_exp_LSTM.network import attentiveLSTM_model 
from models.attention_exp_LSTM.network import SpaceActOptim_model

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
import matplotlib.pyplot as plt

# %%
#load dataset object file
with (open('data/dataset_class.pkl', "rb")) as openfile:
    dataset = pickle.load(openfile)

dataset.activitivity_encoding_mode = 2 #or any value
# Splitting the dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=False )

train_dataloader = DataLoader(train_data, batch_size=100, shuffle=False) #!!! useful to shuffle?
val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)
whole_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# %% [markdown]
# Now that we have seen how this assignment process works, lets put it all together and train a model.

# %% [markdown]
# We'll start loading the already trained y=0 regressor.

# %% [markdown]
# We have to modify this function to be able to generate the prediction graph with the modified espai occupation

# %%
def predict_consumption_opti(model,opti_model,dataloader,y_idx):
    real = []
    pred = []
    h, c = model.init_hidden(dataloader.batch_size) # Start with a new state in each batch            
    h = h.to(device)
    c = c.to(device)
    real_assigns = []
    pred_assigns = []
    h, c = model.init_hidden(1) # Start with a new state in each batch   
    print("running predictions...")   
    for batch, data in tqdm(enumerate(dataloader),total=len(dataset)):
        with torch.no_grad():
            #current batch size size
            ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
            y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now
            
            gener_espai_enc = opti_model(ocu_emb,general_data)
            
            #save assigns for further analysis
            real_assigns += list(espai_enc.view(espai_enc.shape[0]*espai_enc.shape[1]).detach().to("cpu").numpy())
            pred_assigns += list(gener_espai_enc.view(gener_espai_enc.shape[0]*gener_espai_enc.shape[1]).detach().to("cpu").numpy())
            
            h = h.to(device)
            c = c.to(device)
            y_pred, h,c= model(ocu_emb, gener_espai_enc, general_data, h, c)
            pred += list(y_pred[:,0,0].to("cpu"))
            real += list(y.to("cpu"))
    
    return real ,pred, real_assigns, pred_assigns

# %%
def predict_consumption(model,dataloader,y_idx):
    real = []
    pred = []
    h, c = model.init_hidden(dataloader.batch_size) # Start with a new state in each batch            
    h = h.to(device)
    c = c.to(device)
    h, c = model.init_hidden(1) # Start with a new state in each batch    
    print("running predictions...")     
    for batch, data in tqdm(enumerate(dataloader),total=len(dataset)):
        with torch.no_grad():
            #current batch size size
            ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
            y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now
            h = h.to(device)
            c = c.to(device)
            y_pred, h,c= model(ocu_emb, espai_enc, general_data, h, c)
            pred += list(y_pred[:,0,0].to("cpu"))
            real += list(y.to("cpu"))
    return real ,pred

# %% [markdown]
# # Y=0

# %%
y_idx = 0

# %%
model_enpr = attentiveLSTM_model(espai_emb_dim=50,hidden_dim=384,lstm_nl=1,nheads=2,attnFCdim=80)
model_enpr.load_state_dict(torch.load(f"models/attention_exp_LSTM/LSTM_attention_regressor_{y_idx}.pt",map_location=torch.device("cpu")))
model_enpr = model_enpr.to(device)
model_enpr.requires_grad_ = False #freeze our model
opti_model = SpaceActOptim_model(hidden_dim=800,nheads=2,attnFCdim=400)
opti_model = opti_model.to(device)

optimizer = torch.optim.Adam(opti_model.parameters(), lr = 0.001) #lr used in that 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=(1 - 0.0001 / 0.001))
num_epochs = 100
best_loss = float('inf')
epoch_loss_tr = []
epoch_loss_vl = []
with wandb.init(project="energy_project_uab", entity = "energy_project_uab") as run:
    run.name = f"opti_model_{y_idx}.pt"
    for epoch in range(num_epochs):
        opti_model.train()
        training_losses = [] # renamed from epoch_losses
        progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch,data in progress_bar:
            ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
            y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now

            optimizer.zero_grad()
            #current batch size size
            b_sz = ocu_emb.shape[0]
            
            #generate espai_enc
            gener_espai_enc = opti_model(ocu_emb,general_data)

            #ic(gener_espai_enc.shape)
            #ic(espai_enc.shape)
            
            #note the dataloader with a batch of 100 when reachs the end expects a batch of 60
            h, c = model_enpr.init_hidden(b_sz) # Start with a new state in each batch            
            h = h.to(device)
            c = c.to(device)
            y_pred, h,c= model_enpr(ocu_emb, gener_espai_enc, general_data, h, c)
            y_pred = y_pred[:,0,0]

            #loss is 1/(difference of consumption)
            loss = 1/(y_pred-y)
            loss = torch.mean(loss)#get mean loss over batches
            loss.backward()
            optimizer.step()    

            training_losses.append(loss.item())
            progress_bar.set_postfix({'Batch Loss': loss.item()})

        average_training_loss = sum(training_losses) / len(training_losses) # renamed from avg_loss
        epoch_loss_tr.append(average_training_loss)
        wandb.log({"opt_Train_loss":average_training_loss})
        #average_training_loss = np.power(dataset.denormalize_values(np.sqrt(average_training_loss),scaler),2)

        opti_model.eval()  
        with torch.no_grad():  
            validation_losses = [] # renamed from val_losses
            for batch in tqdm(val_dataloader, desc='Validation'):
                ocu_emb, espai_enc, general_data = data["ocu_ber_emb"].float().to(device) ,data["espai_enc"].float().to(device) ,data["general_data"].float().to(device)
                y = data["y"][:,y_idx].float().to(device) #we'll do one counter for now

                optimizer.zero_grad()
                 #current batch size size
                b_sz = ocu_emb.shape[0]
                #generate espai_enc
                gener_espai_enc = opti_model(ocu_emb,general_data)

                #note the dataloader with a batch of 100 when reachs the end expects a batch of 60
                h, c = model_enpr.init_hidden(b_sz) # Start with a new state in each batch            
                h = h.to(device)
                c = c.to(device)
                y_pred, h,c= model_enpr(ocu_emb, gener_espai_enc, general_data, h, c)
                y_pred = y_pred[:,0,0]

                #loss is 1/(difference of consumption)
                loss = 1/(y_pred-y)
                loss = torch.mean(loss)#get mean loss over batches

                validation_losses.append(loss.item())

            average_validation_loss = sum(validation_losses) / len(validation_losses) # renamed from avg_val_loss
            #average_validation_loss = np.power(dataset.denormalize_values(np.sqrt(average_validation_loss),scaler),2)
            epoch_loss_vl.append(average_validation_loss)
            wandb.log({"opt_Val_loss":average_validation_loss})
        
        scheduler.step() #change lr
        print(f"lr:{scheduler.get_last_lr()}")#print current lr

        if average_validation_loss < best_loss:
            best_loss = average_training_loss
            torch.save(opti_model.state_dict(), f'models/attention_exp_LSTM/opti_model_{y_idx}.pt')
            wandb.save(f'models/attention_exp_LSTM/opti_model_{y_idx}.pt')
            print(f'Model saved at models/attention_exp_LSTM/opti_model_{y_idx}.pt')
        
    #if epoch == num_epochs-1: #if last epoch just finished
    if True:
        print("Running whole dataset prediction")
        real, pred = predict_consumption(model_enpr,whole_dataloader,y_idx)
        real ,pred_impr, real_assigns, pred_assigns = predict_consumption_opti(model_enpr,opti_model,whole_dataloader,y_idx)
        #generate hours index
        hours = []
        for i in range(len(dataset)):
            hours += [i]*34 #number of activities
        
        df = pd.DataFrame(columns=["hour", "Real", "Assigned"])
        df["hour"] = hours
        df["Real"] = real_assigns
        df["Assigned"] = pred_assigns
        df.to_csv(f"opti_model_{y_idx}.csv")
        wandb.save(f"opti_model_{y_idx}.csv")

        real = [v.item() for v in real]
        pred = [v.item() for v in pred]
        pred_impr = [v.item() for v in pred_impr]
        scaler = dataset.column_scalers[dataset.target_labels[y_idx]]
        real_sc = dataset.denormalize_values(real,scaler)
        pred_sc = dataset.denormalize_values(pred,scaler)
        pred_impr =dataset.denormalize_values(pred_impr,scaler)
        res_sc = pred_impr-real_sc
        # Plotting the main data
        plt.subplot(3, 1, 1)  # Create a subplot with 2 rows and 1 column, and select the first subplot
        plt.plot(real_sc, label='Real')  # Add a label for the real data
        plt.plot(pred_sc, label='Predicted')  # Add a label for the predicted data

        # Plotting the residual data
        plt.subplot(3, 1, 2)  # Select the second subplot
        plt.plot(real_sc, label='Real')  # Add a label for the real data
        plt.plot(pred_impr, label='Improved')  # Add a label for the predicted data

        plt.subplot(3,1,3)
        plt.plot(res_sc, label='Residual impr-real')  # Add a label for the residual data

        wandb.log({"Optimizing KWh hourly in 2022": plt})

                    # Display the plot
        plt.show()

    wandb.finish()






