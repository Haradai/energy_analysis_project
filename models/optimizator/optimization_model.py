import torch.nn as nn
import torch.nn.init as init
import torch

BERT_DIM_EMB = 768
TIME_CLIMATE_DIM = 21

class AttentionBlock(nn.Module):
    def __init__(self, key_dim, val_dim, query_dim, hidden_dim, num_heads,attnFCdim):
        super(AttentionBlock, self).__init__()
        self.key_gen = nn.Sequential(
            nn.Linear(key_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.ReLU(),
        )

        self.val_gen = nn.Sequential(
            nn.Linear(val_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.ReLU(),
        )
        
        self.query_gen = nn.Sequential(
            nn.Linear(query_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.Softmax(dim=-1),
        )
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    
    def forward(self,keys,values,queries):
        key = self.key_gen(keys) #generate key with FC key is directly the embedding
        value = self.val_gen(values) #generate value with FC
        query = self.query_gen(queries) #generate query with FC
        output, _ =  self.multihead_attn(key=key, value=value, query=query)
        return output


class attentive_optimizer(nn.Module):
    def __init__(self, hidden_dim,lstm_nl,nheads,attnFCdim):
        super(attentive_optimizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_nl = lstm_nl
        self.nheads = nheads

        self.selfattACT =  AttentionBlock(key_dim=BERT_DIM_EMB, val_dim=BERT_DIM_EMB, query_dim=BERT_DIM_EMB, hidden_dim=hidden_dim, num_heads=nheads,attnFCdim=attnFCdim)
        self.selfattKLIM =  AttentionBlock(key_dim=TIME_CLIMATE_DIM, val_dim=TIME_CLIMATE_DIM, query_dim=TIME_CLIMATE_DIM, hidden_dim=hidden_dim, num_heads=nheads,attnFCdim=attnFCdim)
        self.selfattOUT =  AttentionBlock(key_dim=hidden_dim, val_dim=hidden_dim, query_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=nheads,attnFCdim=attnFCdim)
        
        self.outFC= nn.Linear(in_features=hidden_dim,out_features=54) #54 different possible espais

        self.softmax = nn.softmax()

    def forward(self,ocu_ber_emb,espai_enc,general_data,h,c):

        #first extend the general data so we have a pair for each class vector
        general_rep = general_data.unsqueeze(1).repeat(1, 34, 1)

        ACTrepr = self.selfattACT(ocu_ber_emb,ocu_ber_emb,ocu_ber_emb)
        KLIMrepr = self.selfattKLIM(general_rep,general_rep,general_rep)
        
        #gate ACT data with climate data
        KLIM_repr
        keys = self.selfattn1(espai_emb,espai_emb,espai_emb)
        
        out = keys + values + queries #+ out

        out1 = out * self.gatingFC_ocu(queries)
        out2 = out * self.gatingFC_esp(keys)

        out = out1 + out2

        out = torch.sum(out,axis=1)

        #create extra to output be a sequence of 1
        out = out.unsqueeze(1)
        
        #pass this as the inital state to a LSTM
        out, (h,c) = self.lstm(out, (h,c))
        out =  self.regressFC(out).float()
        #out = out.unsqueeze(1) #extra dim to reuse train and predict function
        return out, h, c
    
    def init_hidden(self,batch_size):
        # Initialize the hidden state and cell state with zeros
        h = torch.zeros(self.lstm_nl, batch_size, self.hidden_dim,dtype=torch.float32)
        c = torch.zeros(self.lstm_nl, batch_size, self.hidden_dim,dtype=torch.float32)
        return h, c

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
