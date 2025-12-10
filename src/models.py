import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import torch.nn.functional as F


from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
torch.manual_seed(1)

import math
from torch.nn.utils import weight_norm



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class TranAD_baseline(nn.Module):
    def __init__(self, feats):
        super(TranAD_baseline, self).__init__()
        self.name = 'TranAD_baseline'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 60
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # 25차원 concat 25차원 -> 50차원 src.shape torch.Size([10, 128, 50])
        src = src * math.sqrt(self.n_feats)  # 5
        src = self.pos_encoder(src)  # after positional src.shape torch.Size([10, 128, 50])
        memory = self.transformer_encoder(src)  # memory.shape :  torch.Size([10, 128, 50])
        tgt = tgt.repeat(1, 1, 2)  # new tgt.shape torch.Size([1, 128, 50])
        return  tgt, memory

    def forward(self, src, tgt):  # src shape torch.Size([10, 128, 25])  tgt shape torch.Size([1, 128, 25])
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)  # c shape torch.Size([10, 128, 25])
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Transformer_many_layer(nn.Module):
    def __init__(self, feats):
        super(Transformer_many_layer, self).__init__()
        self.name = 'Transformer_many_layer'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 3)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 3)
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src, save_attention=False):
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        b = self.transformer_decoder1(src, memory)
        x1 = self.fcn2(b)
        
        return x1
    
    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer_layer3(nn.Module):
    def __init__(self, feats):
        super(Transformer_layer3, self).__init__()
        self.name = 'Transformer_layer3'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 3)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src, save_attention=False):
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        b = self.transformer_decoder1(src, memory)
        x1 = self.fcn2(b)
        
        return x1
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer_layer2(nn.Module):
    def __init__(self, feats):
        super(Transformer_layer2, self).__init__()
        self.name = 'Transformer_layer2'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src, save_attention=False):
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        b = self.transformer_decoder1(src, memory)
        x1 = self.fcn2(b)
        
        return x1
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer_final(nn.Module):
    def __init__(self, feats):
        super(Transformer_final, self).__init__()
        self.name = 'Transformer_final'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src, save_attention=False):
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        b = self.transformer_decoder1(src, memory)
        x1 = self.fcn2(b)
        
        return x1
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    



class Transformer(nn.Module):
    def __init__(self, feats):
        super(Transformer, self).__init__()
        self.name = 'Transformer'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(128, 0.1, self.n_window)
        self.multi_scale_fusion = MultiScaleFeatureFusion(2 * feats)

        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
        self.fcn2 = nn.Sequential(nn.Linear(128, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=128)

    def forward(self, src):  # src shape torch.Size([10, 128, 25])  tgt shape torch.Size([1, 128, 25])
        # Phase 1 - Without anomaly scores
        # src shape torch.Size([10, 128, 25])
        #print("1 src.shape", src.shape)
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        #print("2 after pos src.shape", src.shape)
        memory = self.transformer_encoder(src)
        #print("3 memory.shape", memory.shape)
        b = self.transformer_decoder1(src, memory)
        #print("4 b.shape", b.shape)
        x1 = self.fcn2(b)
        #print("5 x1.shape", x1.shape)
        
        return x1

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer2(nn.Module):
    def __init__(self, feats):
        super(Transformer2, self).__init__()
        self.name = 'Transformer2'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 60
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)
        self.multi_scale_fusion = MultiScaleFeatureFusion(2 * feats)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src):  # src shape torch.Size([10, 128, 25])  tgt shape torch.Size([1, 128, 25])
        # Phase 1 - Without anomaly scores
        # src shape torch.Size([10, 128, 25])
        #print("1 src.shape", src.shape)
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        #print("2 after pos src.shape", src.shape)
        memory = self.transformer_encoder(src)
        #print("3 memory.shape", memory.shape)
        b = self.transformer_decoder1(src, memory)
        #print("4 b.shape", b.shape)
        x1 = self.fcn2(b)
        #print("5 x1.shape", x1.shape)
        
        return x1
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer3(nn.Module):
    def __init__(self, feats):
        super(Transformer3, self).__init__()
        self.name = 'Transformer3'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(64, 0.1, self.n_window)
        self.multi_scale_fusion = MultiScaleFeatureFusion(2 * feats)

        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
        self.fcn2 = nn.Sequential(nn.Linear(64, feats))

        self.value_embedding = TokenEmbedding(c_in=15, d_model=64)

    def forward(self, src):  # src shape torch.Size([10, 128, 25])  tgt shape torch.Size([1, 128, 25])
        # Phase 1 - Without anomaly scores
        # src shape torch.Size([10, 128, 25])
        #print("1 src.shape", src.shape)
        src = src.permute(1, 0, 2)
        src = self.value_embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        #print("2 after pos src.shape", src.shape)
        memory = self.transformer_encoder(src)
        #print("3 memory.shape", memory.shape)
        b = self.transformer_decoder1(src, memory)
        #print("4 b.shape", b.shape)
        x1 = self.fcn2(b)
        #print("5 x1.shape", x1.shape)
        
        return x1


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Add attention weights capture
        src2, self_attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, self_attn_weights  # Return attention weights
    

    
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Add attention weights capture
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights  # Return attention weights


    
class TranAD_attention(nn.Module):
    def __init__(self, feats):
        super(TranAD_attention, self).__init__()
        self.name = 'TranAD_attention'
        self.batch = 64
        self.lr = lr
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.multi_scale_fusion = MultiScaleFeatureFusion(2 * feats)

        encoder_layers = CustomTransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = CustomTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = CustomTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

        # For storing attention weights
        self.attention_weights = []

    def encode(self, src, c, tgt, save_attention=False):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory, attn_weights = self.transformer_encoder(src)
        
        # Save attention weights only if save_attention is True
        if save_attention:
            self.attention_weights.append(attn_weights.cpu())
        #print(len(self.attention_weights))
        #print(self.attention_weights[0].shape)
        
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory


    def forward(self, src, tgt, save_attention=False):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        tgt1, memory1 = self.encode(src, c, tgt, save_attention=save_attention)
        decoder_output1, _, _ = self.transformer_decoder1(tgt1, memory1)  # Extract the first value (Tensor)
        x1 = self.fcn(decoder_output1)  # Pass only the Tensor to self.fcn

        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        tgt2, memory2 = self.encode(src, c, tgt, save_attention=save_attention)
        decoder_output2, _, _ = self.transformer_decoder2(tgt2, memory2)  # Extract the first value (Tensor)
        x2 = self.fcn(decoder_output2)  # Pass only the Tensor to self.fcn

        return x1, x2


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Compute attention
        attn_output, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        # Add & Norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # Return src and attention weights
        return src, attn_weights
    

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Decoder Self-Attention
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Encoder-Decoder Cross-Attention
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, self_attn_weights, cross_attn_weights



class Transformer_attention(nn.Module):
    def __init__(self, feats):
        super(Transformer_attention, self).__init__()
        self.name = 'Transformer_attention'
        self.batch = 64
        self.lr = 1e-3  # 학습률
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(128, 0.1, self.n_window)

        # Transformer Components
        encoder_layers = CustomTransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = CustomTransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = CustomTransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
        self.fcn2 = nn.Sequential(nn.Linear(128, feats))

        # Token Embedding
        self.value_embedding = TokenEmbedding(c_in=15, d_model=128)

        # Attention Weights Storage
        self.encoder_attention_weights = []
        self.decoder_self_attention_weights = []
        self.decoder_cross_attention_weights = []

    def forward(self, src, save_attention=False):
        # Step 1: Embedding and Positional Encoding
        src = src.permute(1, 0, 2)  # [batch_size, window_size, feats] -> [window_size, batch_size, feats]
        src = self.value_embedding(src)
        src = self.pos_encoder(src)

        # Step 2: Transformer Encoder
        for layer in self.transformer_encoder.layers:
            src, attn_weights = layer(src)
            if save_attention:
                self.encoder_attention_weights.append(attn_weights.detach().cpu())  # Save encoder attention

        # Step 3: Transformer Decoder
        memory = src


        tgt = memory  # Initial decoder input
        for layer in self.transformer_decoder1.layers:
            tgt, self_attn_weights, cross_attn_weights = layer(tgt, memory)
            if save_attention:
                self.decoder_self_attention_weights.append(self_attn_weights.detach().cpu())
                self.decoder_cross_attention_weights.append(cross_attn_weights.detach().cpu())

        # Step 4: Fully Connected Layer
        output = self.fcn2(tgt)

        # Reshape back to [batch_size, window_size, feats]
        return output.permute(1, 0, 2)




# class Transformer(nn.Module):
#     def __init__(self, feats):
#         super(Transformer, self).__init__()
#         self.name = 'Transformer'
#         self.batch = 64
#         self.lr = lr
#         self.n_feats = feats
#         self.n_window = 120
#         self.n = self.n_feats * self.n_window
#         self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
#         self.multi_scale_fusion = MultiScaleFeatureFusion(2 * feats)

#         encoder_layers = TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
#         decoder_layers1 = TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
#         self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
#         decoder_layers2 = TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
#         self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
#         self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
#         self.fcn2 = nn.Sequential(nn.Linear(feats, feats))

#         self.value_embedding = TokenEmbedding(c_in=15, d_model=128)

#     def forward(self, src):  # src shape torch.Size([10, 128, 25])  tgt shape torch.Size([1, 128, 25])
#         # Phase 1 - Without anomaly scores
#         # src shape torch.Size([10, 128, 25])
#         #print("1 src.shape", src.shape)
#         src = src.permute(1, 0, 2)
#         src = self.pos_encoder(src)
#         src = src.permute(1, 0, 2)
#         #print("2 after pos src.shape", src.shape)
#         memory = self.transformer_encoder(src)
#         #print("3 memory.shape", memory.shape)
#         b = self.transformer_decoder1(src, memory)
#         #print("4 b.shape", b.shape)
#         x1 = self.fcn2(b)
#         #print("5 x1.shape", x1.shape)
        
#         return x1







#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#아래는 attention 뽑을수있는 Transformer autoencoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Compute attention
        attn_output, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        # Add & Norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # Return src and attention weights
        return src, attn_weights


class Transformer_attention(nn.Module):
    def __init__(self, feats):
        super(Transformer_attention, self).__init__()
        self.name = 'Transformer_attention'
        self.batch = 64
        self.lr = 1e-3  # 학습률
        self.n_feats = feats
        self.n_window = 120
        self.n = self.n_feats * self.n_window

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(128, 0.1, self.n_window)

        # Transformer Components
        encoder_layers = CustomTransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(feats, feats), nn.Sigmoid())
        self.fcn2 = nn.Sequential(nn.Linear(128, feats))

        # Token Embedding
        self.value_embedding = TokenEmbedding(c_in=15, d_model=128)

        # Attention Weights Storage
        self.encoder_attention_weights = []

    def forward(self, src, save_attention=False):

        # Step 1: Embedding and Positional Encoding
        src = src.permute(1, 0, 2)  # [batch_size, window_size, feats] -> [window_size, batch_size, feats]
        src = self.value_embedding(src)
        src = self.pos_encoder(src)

        # Step 2: Transformer Encoder
        encoder_attention_maps = []
        #src, attn_weights = self.transformer_encoder(src)
        for layer in self.transformer_encoder.layers:
            src, attn_weights = layer(src)
            if save_attention:
                self.encoder_attention_weights.append(attn_weights.detach().cpu())




        # Step 3: Transformer Decoder
        memory = src
        output = self.transformer_decoder1(src, memory)

        # Step 4: Fully Connected Layer
        output = self.fcn2(output)

        # Reshape back to [batch_size, window_size, feats]
        return output.permute(1, 0, 2)

    def get_attention_maps(self):
        return {"encoder_attention_weights": self.encoder_attention_weights}