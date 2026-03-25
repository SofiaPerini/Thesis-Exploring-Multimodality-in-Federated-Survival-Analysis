
import torch
import numpy as np 
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from src.models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb

import math
import pandas as pd

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    Used for tokenization of pathways from transcriptomics (omics data)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class SurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes = [100, 200, 300, 400, 500, 600],
        wsi_embedding_dim = 1536,    #### changed here from original 1024
        dropout = 0.1,
        num_classes = 4,
        wsi_projection_dim = 256,
        omic_names = [],
        omic_hidden_dim = 256,
        ):
        super(SurvPath, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout

        #---> omics preprocessing for captum - omics_names only used by captum? 
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(          
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),   ### Applies an affine linear transformation to the incoming data: y=xAT+by=xAT+b.
        )     # A sequential container. Modules will be added to it in the order they are passed in the constructor. 
              # The forward() method of Sequential accepts any input and forwards it to the first module it contains. 
              # It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
              # The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the Sequential applies to each of the modules it stores

        #---> omics props
        self.init_per_path_model(omic_sizes, omic_hidden_dim, wsi_projection_dim)   # returns ModuleList of modules to manage the pathways

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig -- forwarding the input given to it, does not change the input
        self.cross_attender = MMAttentionLayer(
            dim = self.wsi_projection_dim,
            dim_head = self.wsi_projection_dim // 2, ## (floor division)
            heads = 1,
            residual = False,
            dropout = 0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        


    def init_per_path_model(self, omic_sizes, omic_hidden_dim, emdebding_dim=256):
        hidden = [omic_hidden_dim, emdebding_dim]   ## [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]    # from input dim to hidden[0] (256) as output dim
            for i, _ in enumerate(hidden[1:]):    ## do it once??
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))  ## add more SNN_block, from 256 dim to 256 dim output
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)  # only a list of modules, not sequential  
    


    def forward(self, **kwargs):   
        r'''
        Define the computation performed at every call. To use the model, we pass it the input data. This executes the model’s forward, along with some background operations
        '''
        
        wsi = kwargs['x_path'] if kwargs.get('is_wsi', True) else None     ## WSI images, name added in code, may be equal to 0 if no wsi data is present
        if kwargs.get('is_omics', True) and kwargs.get('x_omic1') is not None :   # if omics data is present
            x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]  ## only omics data
            #print('x_omic: ', x_omic)
        else:
            x_omic = None
        # if num_pathways = 3, returns kwargs['x_omic1'], kwargs['x_omic2'], kwargs['x_omic3']
        
        mask = None
        return_attn = kwargs["return_attn"]   # False
        
        device_i = kwargs['device'] if 'device' in kwargs else torch.device('cpu')

        #print(f'wsi shape: {wsi.shape if wsi is not None else None}')   wsi shape: torch.Size([1, 64, 1536])
        #print(f'x_omic shapes: {[x.shape for x in x_omic] if x_omic is not None else None}')   x_omic shapes: [torch.Size([4096]), torch.Size([2404]), torch.Size([191]), torch.Size([4096])]
        
        #---> get pathway embeddings
        if x_omic is None:   # if no omics data is present
            # fill with zeroes
            h_omic_bag = torch.zeros((1, self.num_pathways, self.wsi_projection_dim)).to(wsi.device)

        # one element of x_omic is all zeroes: set corresponding h_omic to zeroes
        elif any(torch.sum(sig_feat)==0 for sig_feat in x_omic):
            h_omic = [self.sig_networks[idx].forward(sig_feat.float()) if torch.sum(sig_feat) != 0 else torch.zeros(self.wsi_projection_dim).to(device_i) for idx, sig_feat in enumerate(x_omic)]
            h_omic_bag = torch.stack(h_omic).unsqueeze(0)
            #print('h_omic if dna missing: ', h_omic)
        else:
            h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature (one for each num_pathways) goes through it's own FC layer
            h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
            #print(f'h_omic_bag shape: {h_omic_bag.shape}')
            #print('h_omic: ', h_omic_bag)
        #---> project wsi to smaller dimension (same as pathway dimension)
        if wsi is None:   # if no wsi data is present
            # fill with zeroes
            wsi_embed = torch.zeros((1, 1, self.wsi_projection_dim)).to(h_omic_bag.device)
        else:
            wsi_embed = self.wsi_projection_net(wsi)
            #print(f'wsi_embed shape: {wsi_embed.shape}')

        #print(f'wsi_embed shape: {wsi_embed.shape}')   wsi_embed shape: torch.Size([1, 64, 64])
        #print(f'h_omic_bag shape: {h_omic_bag.shape}')   h_omic_bag shape: torch.Size([1, 4, 64])
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)  ## concatenate
        #if kwargs.get('is_omics', True)==False or kwargs.get('is_wsi', True)==False:
        #print(tokens)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.to_logits(embedding)

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits
        


        
    def captum(self, omics_0 ,omics_1 ,omics_2 ,omics_3, wsi):
        
        #---> unpack inputs
        mask = None
        return_attn = False
        
        omic_list = [omics_0 ,omics_1 ,omics_2 ,omics_3 ]

        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_list)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic, dim=1)  #.unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        
        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)
        #print(f'wsi_embed shape: {wsi_embed.shape}')
        #print(f'h_omic_bag shape: {h_omic_bag.shape}')
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return risk