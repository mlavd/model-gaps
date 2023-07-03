# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args, class_weights):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.class_weights = class_weights
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss = (
                torch.log(prob[:,0] + 1e-10) * labels * self.class_weights[1] + # Positive
                torch.log((1 - prob)[:,0] + 1e-10) * (1 - labels) * self.class_weights[0] # Negative
            )
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
        
 