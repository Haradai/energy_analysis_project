from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import torch
import numpy as np
import pandas as pd
import pickle
import datetime

class energyProject_dataset(Dataset):
    def __init__(self,dataset_pth,occupacio_pth,bert_embeddings_pkl_pth,pca_pkl_pth=None):
        self.df = pd.read_csv(dataset_pth)
        with (open(bert_embeddings_pkl_pth, "rb")) as openfile:
            self.bert_embeddings = pickle.load(openfile)
        self.activitivity_encoding_mode = 0
        
        #order of values in target tensor will follow this
        self.target_labels = ["Q-Enginyeria (Cos Central) [kWh] [Q-Enginyeria]","Q-Enginyeria (Espina 4) [kWh] [Q-Enginyeria]","Q-Enginyeria (Química) [kWh] [Q-Enginyeria]"]
       
        #load occupation data
        self.occupation_df = pd.read_csv(occupacio_pth)
        #we'll remove entries without date
        self.occupation_df = self.occupation_df[self.occupation_df["Data inicial"] != " "]

        #convert date string to be in the form y-m-d instead of d/m/y
        #convert hour data to datetime object so we can compare them
        for i, row in self.occupation_df.iterrows(): 
            self.occupation_df.loc[i]["Data inicial"] =  datetime.datetime.strptime(self.occupation_df.loc[i]["Data inicial"], "%d/%m/%Y").strftime("%Y-%m-%d")
            self.occupation_df.loc[i]["Hora inicial"] = datetime.datetime.strptime(self.occupation_df.loc[i]["Hora inicial"] ,"%H:%M").time()
            self.occupation_df.loc[i]["Hora final"] = datetime.datetime.strptime(self.occupation_df.loc[i]["Hora final"] ,"%H:%M").time()
        
        self.ocup_vocab = list(set(self.occupation_df["Activitat"]))
        self.espais_vocab = list(set(self.occupation_df["Espai"]))
        #Add a padding occupation
        self.espais_vocab.append("NO ESPAI")

        #Normalize all climate data to be between 0-1
        #we'll keep their scaler objects so we can transform their values back to original
        #and not loose meaning.
        self.column_scalers = {}
        columns_to_process = ['winddirDegree', 'precipMM', 'visibility', 'WindChillC',
       'humidity', 'pressure','windspeedMiles', 'uvIndex', 'DewPointC',
       'FeelsLikeC', 'tempC','weatherCode','HeatIndexC', 'WindGustKmph', 'cloudcover',
       'windspeedKmph','Q-Enginyeria (Cos Central) [kWh] [Q-Enginyeria]',
       'Q-Enginyeria (Química) [kWh] [Q-Enginyeria]',
       'Q-Enginyeria (Espina 4) [kWh] [Q-Enginyeria]'] #normalze also target
     
        for col in columns_to_process:
            scaler, values = self.normalize_values(self.df[col])
            self.df[col] = values
            self.column_scalers[col] = scaler
        
        #If we want this dataset to work with batches in modes different than 0 and 1.
        #We need to know what is the maximum number of activities at the same time so we can
        #padd the samples smaller. We need the batch to have the same shape samples every time.

        #we'll use that we are iterating throught this to compute all the one_hot vectors of the encodings
        #and fit a PCA object so we can have it with dimensionality reduction.

        self.max_ocu_lenght = 0
        self.all_one_hots = []
        for i, row in self.df.iterrows():
            day2day_ocu = self.activty_class_perT(row["date"],row["time"])
            
            #find largest occupation size per hour
            day2day_ocu_l = len(day2day_ocu)
            if day2day_ocu_l > self.max_ocu_lenght:
                self.max_ocu_lenght = day2day_ocu_l
                
            if pca_pkl_pth == True: #if we have to calculate the PCA
                #compute one_hot vectors of occupation at this time
                self.all_one_hots.append(self.activity_class_one_hot(day2day_ocu))
        
        if (pca_pkl_pth == True): #nan value, recalculate and save file
            self.all_one_hots = np.array(self.all_one_hots)
            #compute PCA on all the one_hot vectors
            self.ocu_one_hot_pca = PCA(n_components=1000) #1000 size output vector (number chosen by hand)
            self.ocu_one_hot_pca.fit(self.all_one_hots)
            # Open a file and use dump()
            with open('data/pca_occupation.pkl', 'wb') as file:
                pickle.dump(self.ocu_one_hot_pca, file)
        else:
            with (open(pca_pkl_pth, "rb")) as openfile:
                self.ocu_one_hot_pca = pickle.load(openfile)
        
        ##free memory by removing unnecessary variables.
        del self.all_one_hots

    def normalize_values(self,x):
        """
        Input a list of values
        Output a sklearn scaler object and the list normalized.
        We need to keep the scaler to be able to re-scale the data back and now what value it is in reality.
        """
        to_scale = np.array(x).reshape(-1, 1) #the library needs this extra dimensions trick to interpret properly
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        x_scaled = min_max_scaler.fit_transform(to_scale)
        return min_max_scaler, x_scaled
    
    def denormalize_values(self,x_n,scaler):
        """
        Given some set of values and a sklearn scaler object
        Transform back the values to their original "space"ArithmeticError
        return: set of values same shape as input
        """
        to_scale = np.array(x_n).reshape(-1, 1)
        return scaler.inverse_transform(to_scale)

    def datetime_enc(self,date, time)->torch.tensor:
        """
        Encodes incoming date and time strings as two values for each that 
        come from infering the index value on a sin function and cos function.
        Its nice beacause we encode the smoothness and circularity of the trigonometric
        functions.

        input <- (date: str, time:str)
        output -> (torch.tensor((1,5)))
        """
        
        date_obj =  datetime.datetime.strptime(date, "%Y-%m-%d")
        
        #encode year as floas
        year_enc = float(date_obj.year)/1000 # divide by 100 to have reasonable value
        
        ##
        ##Encoding: (sin, cos) value for each day month
        ##
        idx_d = date_obj.timetuple().tm_yday #day of the year number
        date_enc= [np.sin((idx_d/365) * 2*np.pi),np.cos((idx_d/365) * 2*np.pi)]  #365 days a yar +1 offset so we don't have negative value
        
        #encode time by hour in the day
        time_enc = [np.sin((time/24) * 2*np.pi),np.cos((time/24) * 2*np.pi)]  #24 hours a day. +1 offset so we don't have negative values
        
        return torch.tensor([year_enc]+date_enc+time_enc) 

    def datetime_dec(self,enc_tens):
        """
        Decodes incoming encoded date and time tensor as the two respective
        date time string values

        input <- (torch.tensor([torch.float,torch.float]))
        output -> date: str, time:str
        """
        #decode year
        year = int(enc_tens[0].item() * 1000)

        #decode date
        penc_date = np.arctan2(enc_tens[1].item(),enc_tens[2].item()) / (2*np.pi) * 365
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(penc_date - 1)
        date = date.strftime("%Y-%m-%d")

        #decode time
        time = np.round((np.arctan2(enc_tens[3].item(),enc_tens[4].item()) / (2*np.pi) * 24)% 24)
        
        return date,time

    def activty_class_perT(self,date,time)->pd.DataFrame:
        """
        Returns slice of the pd Dataframe of activities active given some date and time
        """
        #filter dataset to see activities that day
        day2day_ocu = self.occupation_df[self.occupation_df["Data inicial"] == date ] 
        h = datetime.time(int(time))
        hour2hour_ocu = day2day_ocu[(day2day_ocu["Hora inicial"] <= h) & (day2day_ocu["Hora final"] > h)]
        return pd.DataFrame(hour2hour_ocu)

    def activity_class_one_hot(self,activities)->np.array:
        """ 
        returns flattenned coocurrence one-hot matrix of activities and classrooms
        """
        occurrence_matrix = np.zeros((len(self.ocup_vocab),len(self.espais_vocab)))
        for i,actv in activities.iterrows(): #iterate found activities
            ocup_idx = self.ocup_vocab.index(actv["Activitat"])
            espais_idx = self.espais_vocab.index(actv["Espai"])
            occurrence_matrix[ocup_idx,espais_idx] = 1
            
        #now flatten the occurrence matrix into a one hot vector
        one_hot = occurrence_matrix.flatten()
        return one_hot
    
    def class_one_hot(self,activitats)->torch.tensor:
        """
        Returns "one hot" encoding of activitat.
        In reality will not be a true one hot but a list of indexes
        that can later on be passed to some embedding layer
        """
        activitats["Espai"]
        one_hot_esp = [self.espais_vocab.index(key) for key in activitats["Espai"]]
        return torch.tensor(one_hot_esp)
       


    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, index):
        """
        This function will return more than one object depending on the mode it is on.
        Activity encoding mode:
            0: I.very large one-hot encoding of all the combinations of classroom and activity concatenated
                with all other features.
               II. target values
            
            1:  I. 0 but first with some PCA applied to reduce dimensionality of 
                the enormous one-hot encoding.
                II. target values
            
            2: returns four objects, 
                I.mean_encoding of activity from bert(as many as activities at time stamp), 
                II. classroom one hot encoding for each activity(as many as activities at time stamp)
                III. All other features at that time stamp
                IV. target values
            
            2.5:  returns four objects
                I. embedding tensor of activity from bert(as many as activities at time stamp), 
                II. classroom one hot encoding for each activity(as many as activities at time stamp)
                III. All other features at that time stamp
                IV. target values
            
            3: returns three objects,
                I. mean encoding of activity, classroom pair through bert (as many as activities at time stamp),
                II. All other features at that time stamp
                III. target values
            
            3.5:  returns four objects
                I. embedding tensor of activity, classroom pair through bert(as many as activities at time stamp), 
                II. classroom one hot encoding for each activity(as many as activities at time stamp)
                III. All other features at that time stamp
                IV. target values
            
            4: returns three objects, raw data thought for model handling.
                I. activities in text form paired with their classroom
                II. All other features at that time stamp
                III. target values
        """

        #get row in df for data to be evaluated:
        row = self.df.iloc[index]

        #First getting the "all other data " features tensor
        #Date-time encoding tensor
        enc_dt_tens = self.datetime_enc(row["date"],row["time"]) #date-time encoding

        #weather data tensor
        weather_tens = torch.tensor(row.drop(["date","time","Q-Enginyeria (Cos Central) [kWh] [Q-Enginyeria]","Q-Enginyeria (Espina 4) [kWh] [Q-Enginyeria]","Q-Enginyeria (Química) [kWh] [Q-Enginyeria]"]))

        #second get the target values tensor
        target_tens = torch.tensor(row[self.target_labels])
        
        #Get activities at given time and date
        activities = self.activty_class_perT(row["date"],row["time"])
        if self.activitivity_encoding_mode <= 1:
            one_hot = self.activity_class_one_hot(activities)
            
            if(self.activitivity_encoding_mode == 0):
                one_hot = torch.tensor(one_hot)
                #return values
                sample = {'x': torch.cat((enc_dt_tens,weather_tens,one_hot),axis=0), 'y': target_tens}
                return sample
            
            if(self.activitivity_encoding_mode == 1):
                one_hot = one_hot.reshape(1, -1) #create extra dimension because this counts as only one sample
                smaller_x = torch.tensor(self.ocu_one_hot_pca.transform(one_hot))[0] #get rid of extra dim
                sample = {'x': torch.cat((enc_dt_tens,weather_tens,smaller_x),axis=0), 'y': target_tens}
                return sample
        
        if self.activitivity_encoding_mode == 2:
            #get bert embeddings
            emb_activ = []
            for i,actv in activities.iterrows(): #iterate found activities   
                emb_activ.append(self.bert_embeddings["ocu_plus_space"]["mean_vect"][actv["Activitat"] + " " + actv["Espai"]])
            emb_activ = torch.tensor(np.array(emb_activ))

            #get espai one-hot
            espai_one_hot = self.class_one_hot(activities)

            #padd with occupation 0 vector so all samples are same shape and espai with ""NO ESPAI"
            if emb_activ.shape[0] < self.max_ocu_lenght: 
                emb_activ = torch.cat((emb_activ,torch.zeros((self.max_ocu_lenght-emb_activ.shape[0],768))),axis=0)
                espai_padd = pd.DataFrame(columns=["Espai"]) 
                espai_padd["Espai"] = ["NO ESPAI"]* (self.max_ocu_lenght-len(espai_one_hot))
                padd_one_hot = self.class_one_hot(espai_padd)
                espai_one_hot = torch.cat((espai_one_hot,padd_one_hot))
                
            sample = {'ocu_ber_emb': emb_activ,'espai_enc':espai_one_hot, "general_data":torch.cat((enc_dt_tens,weather_tens),axis=0), 'y': target_tens}
            return sample
        
        if self.activitivity_encoding_mode == 2.5:
            assert "NOT IMPLEMENTED YET"
            pass
            
        if self.activitivity_encoding_mode == 3:
            emb_activ = []
            for i,actv in activities.iterrows(): #iterate found activities   
                emb_activ.append(self.bert_embeddings["ocu_plus_space"]["mean_vect"][actv["Activitat"] + " " + actv["Espai"]])
            emb_activ = torch.tensor(np.array(emb_activ))

            if emb_activ.shape[0] < self.max_ocu_lenght: #padd with occupation 0 vector so all samples are same shape
                emb_activ = torch.cat((emb_activ,torch.zeros((self.max_ocu_lenght-emb_activ.shape[0],768))),axis=0)
            
            #return values
            sample = {'activ': torch.tensor(emb_acti), 'general_data':torch.cat((enc_dt_tens,weather_tens),axis=0), 'y': target_tens}
            return sample
        
        if self.activitivity_encoding_mode == 3.5:
            assert "NOT IMPLEMENTED YET"
            #errors to solve have to recalculate bert passing
            emb_acti = []
            for i,actv in activities.iterrows(): #iterate found activities   
                emb_acti.append(self.bert_embeddings["ocu_plus_space"]["h_states"][actv["Activitat"] + " " + actv["Espai"]])
                print(self.bert_embeddings["ocu_plus_space"]["h_states"][actv["Activitat"] + " " + actv["Espai"]].shape)
            emb_acti = torch.tensor(emb_acti)
            
            print(emb_acti.shape)
            #return values
            return torch.tensor(emb_acti), torch.cat((enc_dt_tens,weather_tens),axis=0) , target_tens

        if self.activitivity_encoding_mode == 4:
            acti = list(activities["Activitat"])
            esp = list(activities["Espai"])

            if len(acti) < self.max_ocu_lenght: #padd with occupation 0 vector so all samples are same shape
                pad = ["NONE"] * (self.max_ocu_lenght-len(acti))
                acti += pad
                esp += pad

            sample = {"activ:":acti, "espai": esp,"general_data":torch.cat((enc_dt_tens,weather_tens),axis=0),'y': target_tens}
            return sample