from transformers import AutoConfig,AutoModel
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch

class CrossAttnHead(nn.Module):
    def __init__(self,d_model,nhead=16,dropout=0.2):
        super(CrossAttnHead, self).__init__()
        self.AttnHead=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.dropout1=nn.Dropout(dropout)
        self.norm1= nn.LayerNorm(d_model)
        self.linear1=nn.Linear(d_model, d_model)
        self.linear2=nn.Linear(d_model, d_model)
        self.dropout2=nn.Dropout(dropout)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x, attention_mask):
        res=x[:,0].unsqueeze(1)
        x, _ = self.AttnHead(x[:,0].unsqueeze(1),x[:,1:],x[:,1:],attention_mask)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=res+x

        x=x.squeeze(1)
        res=x
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)

class ResidualLSTM(nn.Module):

    def __init__(self, d_model, rnn='GRU'):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        x=x.permute(1,0,2)
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x = self.linear1(x)
        # x=self.dropout1(x)
        # x=self.norm1(x)
        # x=F.relu(self.linear1(x))
        # x=self.linear2(x)
        # x=self.dropout2(x)
        x=res+x
        x=x.permute(1,0,2)
        return self.norm2(x)



class SlidingWindowTransformerModel(nn.Module):
    def __init__(self,DOWNLOADED_MODEL_PATH, nclass, rnn='GRU', window_size=512, edge_len=64, no_backbone=False):
        super(SlidingWindowTransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')
        self.no_backbone=no_backbone
        if no_backbone:
            pass
        else:
            self.backbone=AutoModel.from_pretrained(
                               DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)

        hidden_state_dimension=self.backbone.embeddings.word_embeddings.embedding_dim

        if rnn=="GRU" or rnn=='LSTM':
            self.lstm=ResidualLSTM(hidden_state_dimension,rnn)
        else:
            self.lstm=ResNet()



        self.classification_head=nn.Linear(hidden_state_dimension,nclass)
        self.window_size=window_size
        self.edge_len=edge_len
        self.inner_len=window_size-edge_len*2

        self.discourse_embedding=nn.Embedding(8,256,padding_idx=0)
        self.downsample=nn.Linear(hidden_state_dimension+256,hidden_state_dimension)

    def forward(self,input_ids,attention_mask,sequence_ids,discourse_type_ids,gather_indices,return_vectors=False,return_transformer_hidden_states=False):



        # print(L)
        # exit()
        #x=self.backbone(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[0]
        #x=self.backbone.embeddings(input_ids)#+0.1*self.discourse_embedding(discourse_type_ids)
        discourse_type_ids=self.discourse_embedding(discourse_type_ids)
        x=input_ids
        # x=torch.cat([x,discourse_type_ids],-1)
        # x=self.downsample(x)

        #x=torch.cat([x,])

        if self.no_backbone==False:
            B,L=input_ids.shape
            if L<=self.window_size:
                x=self.backbone(x,attention_mask=attention_mask,return_dict=False)[0]
                #pass
            else:
                #print("####")
                #print(input_ids.shape)
                segments=(L-self.window_size)//self.inner_len
                if (L-self.window_size)%self.inner_len>self.edge_len:
                    segments+=1
                elif segments==0:
                    segments+=1
                x_new=self.backbone(x[:,:self.window_size],attention_mask=attention_mask[:,:self.window_size],return_dict=False)[0]
                # print(x_new.shape)
                # exit()

                for i in range(1,segments+1):
                    start=self.window_size-self.edge_len+(i-1)*self.inner_len
                    end=self.window_size-self.edge_len+(i-1)*self.inner_len+self.window_size
                    end=min(end,L)
                    x_next=x[:,start:end]
                    mask_next=attention_mask[:,start:end]
                    x_next=self.backbone(x_next,attention_mask=mask_next,return_dict=False)[0]
                    #L_next=x_next.shape[1]-self.edge_len,
                    if i==segments:
                        x_next=x_next[:,self.edge_len:]
                    else:
                        x_next=x_next[:,self.edge_len:self.edge_len+self.inner_len]
                    #print(x_next.shape)
                    x_new=torch.cat([x_new,x_next],1)
                x=x_new
                #print(start,end)
        #print(x.shape)
            if return_transformer_hidden_states:
                transformer_hidden_states=x

            # print(x.shape)
            # exit()

            # x=torch.cat([x,discourse_type_ids],-1)
            # x=self.downsample(x)

            #x=self.lstm(x)

            #x=self.classification_head(x).squeeze(-1)

            pooled_outputs=[]
            if return_vectors:
                vectors=[]
            for i in range(len(x)):
                #n_discourses=gather_indices[i].max()+1
                # unique_gather_indices=torch.unique_consecutive(gather_indices[i])
                # unique_gather_indices=unique_gather_indices[unique_gather_indices!=-1]
                #
                # #print(unique_gather_indices)
                #
                # for j in unique_gather_indices:
                n_discourses=gather_indices[i].max()+1
                tmp=[]
                for j in range(n_discourses):


                    vector=x[i][gather_indices[i]==j]
                    if return_vectors:
                        vectors.append(self.classification_head(vector))
                    mean_vector=vector.mean(0)
                    #max_vector,_=vector.max(0)
                    # print(max_vector)
                    # exit()
                    #pooled=torch.cat([mean_vector,max_vector],-1)
                    #pooled=mean_vector
                    tmp.append(mean_vector)
                    #pooled_outputs.append(pooled)
                tmp=torch.stack(tmp)
                tmp=self.lstm(tmp.unsqueeze(0))
                pooled_outputs.append(tmp.squeeze(0))


            #exit()
            pooled_outputs=torch.cat(pooled_outputs)
            x=pooled_outputs
            x=self.classification_head(x).squeeze(-1)


            # if return_vectors:
            #     vectors=torch.stack(vectors,0)
            #     vectors=self.classification_head(vectors)

            #seq_mask=(sequence_ids==0)

            #sum_L=(x*seq_mask.unsqueeze(-1)).sum(1)
            #sum_mask=seq_mask.sum(1)

            # print(sum_L.shape)
            # print(sum_mask.shape)
            # exit()


            #pooled=sum_L/sum_mask.unsqueeze(-1)



            #x=self.classification_head(x)
        else:
            transformer_hidden_states=input_ids
            x=self.lstm(transformer_hidden_states)
            x=self.classification_head(x)

        if return_vectors:
            return x,vectors
        else:
            return x

        # if return_transformer_hidden_states:
        #     return x, transformer_hidden_states
        # else:
        #     return x#, BIO_output
