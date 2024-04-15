import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = get_device()

@dataclass
class Config :
    model_d: int
    max_sequence_len: int
    ger_vocab_size: int
    eng_vocab_size : int
    n_layers : int
    n_heads : int
    eng_to_index : dict
    ger_to_index : dict
    start_token : str
    end_token : str
    pad_token: str
    dropout_p : int = 0.1
    hidden_size : int = 2048

class Tokenizer(nn.Module) :
      def __init__(self,config,layer_type) :
            super(Tokenizer,self).__init__()
            self.vocab_size = config.eng_vocab_size if layer_type == "encoder" else config.ger_vocab_size
            self.embedding = nn.Embedding(self.vocab_size,config.model_d)
            self.max_sequence_length = config.max_sequence_len
            self.language_to_index =  config.eng_to_index if layer_type == "encoder" else config.ger_to_index
            self.start_token = config.start_token
            self.end_token = config.end_token
            self.pad_token = config.pad_token
            self.dropout = nn.Dropout(p=config.dropout_p)
            self.positional_embedding = Positional_embedding(config)
      def batch_tokenization(self,batch,is_starttoken,is_endtoken) :
              def sentence_tokenize(sentence,is_starttoken,is_endtoken) :
                      sentence_to_index = []
                      for tokenIdx,token in enumerate(list(sentence)) :
                              if tokenIdx + int(is_starttoken) + int(is_endtoken) + 1 >= self.max_sequence_length :
                                          break
                              if token in self.language_to_index:
                                      sentence_to_index.append(self.language_to_index[token] )
                              else :
                                      sentence_to_index.append(self.language_to_index['<unk>'])
                      if is_starttoken :
                            sentence_to_index.insert(0,self.language_to_index[self.start_token])
                      if is_endtoken :
                            sentence_to_index.append(self.language_to_index[self.end_token])

                      for _ in range(len(sentence_to_index),self.max_sequence_length):
                                  sentence_to_index.append(self.language_to_index[self.pad_token])
                      return torch.tensor(sentence_to_index).to(device)

              sentence_batch = []
              for sentence_idx in range(len(batch)) :
                    sentence_batch.append(sentence_tokenize(batch[sentence_idx],is_starttoken,is_endtoken))

              sentence_batch = torch.stack(sentence_batch)
              return sentence_batch.to(device)
      def forward(self,x,is_starttoken,is_endtoken) :
            # (batch,vocab_size,embed_d)
            print("============== Tokenization ===============")
            x = self.batch_tokenization(x,is_starttoken,is_endtoken)
            x = self.embedding(x)
            pos = self.positional_embedding(x)
            x = self.dropout(x+pos)
            return x


class Positional_embedding(nn.Module):
      def __init__(self,config) :
            super(Positional_embedding,self).__init__()
            self.model_d = config.model_d
            self.max_len = config.max_sequence_len
            self.positional_embedding = torch.zeros((self.max_len,self.model_d)).to(device)
            pos = torch.arange(0,self.max_len,dtype=torch.float).unsqueeze(1)
            div = torch.pow(1000,torch.arange(0,self.model_d,2,dtype=torch.float)/self.model_d)
            self.positional_embedding[:,0::2] = torch.sin(pos/div)
            self.positional_embedding[:,1::2] = torch.cos(pos/div)
      def forward(self,x) :
            print("============ positional embedding ==========",x.size(),self.positional_embedding.size())
            x = x +  self.positional_embedding
            return x

class LayerNorm(nn.Module):
      def __init__(self,epsilon=1e-6):
              super(LayerNorm,self).__init__()
              self.epsilon = epsilon

      def forward(self,x) :
              print("============== LayerNormalization ===============", x.size())
              batch_size,seq_length,model_d = x.shape
              gamma = nn.Parameter(torch.ones(model_d).to(device))
              beta = nn.Parameter(torch.zeros(model_d).to(device))
              mean = x.mean(-1,keepdim=True)
              var = x.var(-1,keepdim=True)
              x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)

              x = gamma * x_normalized + beta
              return x
      
class MultiHeadAttention(nn.Module) :
      def __init__(self,config) :
                  super(MultiHeadAttention,self).__init__()
                  assert config.model_d % config.n_heads == 0  , f"{config.model_d} should be divisible by {config.n_heads}"
                  self.max_sequence_len = config.max_sequence_len
                  self.model_d = config.model_d
                  self.heads_num = config.n_heads
                  self.qkv_d = self.model_d // self.heads_num
                  self.queryP = nn.Linear(self.model_d,self.model_d)
                  self.valueP = nn.Linear(self.model_d,self.model_d)
                  self.keyP = nn.Linear(self.model_d,self.model_d)
                  self.out = nn.Linear(self.model_d,self.model_d)
      def attention(self,q,k,v,mask=None) :
             dk = torch.tensor(k.shape[-1],dtype=torch.float32)
             energy = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(dk)
             if mask != None  :
                        print("________Masking__________")
                        energy = torch.permute(energy,(1,0,2,3))
                        print("energy shape",energy.shape)
                        print("mask shape",mask.shape)
                        energy =  energy.masked_fill(mask != 0, float('-1e9'))
                        energy = torch.permute(energy,(1,0,2,3))

             return torch.matmul(torch.softmax(energy,dim=-1),v)
      def forward(self,x,mask=None):
              """
               first we create key,query and value using a linear projection using a 1 fully connected layer
               The size of these tensors is (input_sequence_length,model_d)
              """
              print("============== MultiHeadAttention ===============")
              query= None
              key = None
              value = None
              batch_size = None
              if isinstance(x,(list,tuple)) :
                   query,key,value = x
                   batch_size = query.size(0)
                   query = self.queryP(query)
                   key = self.keyP(key)
                   value = self.valueP(value)

              else :
                   batch_size = x.size(0)
                   query = self.queryP(x)
                   key = self.keyP(x)
                   value = self.valueP(x)

              """
               we add another dimension for heads now the tensors size is (heads_num,input_sequence,model_d)
               calculte attention for each head independently and in parallel
              """
              query = query.view(batch_size,self.heads_num,self.max_sequence_len,self.qkv_d)
              key = key.view(batch_size, self.heads_num,self.max_sequence_len, self.qkv_d)
              value = value.view(batch_size, self.heads_num,self.max_sequence_len, self.qkv_d)
              attention = self.attention(query, key, value, mask)
              attention = attention.view(batch_size, self.max_sequence_len, self.model_d)
              out = self.out(attention)
              return out

class FeedForward(nn.Module) :
        def __init__(self,input_size,output_size,hidden_size,dropout_p=0.1) :
                  super(FeedForward,self).__init__()
                  self.fc1 =  nn.Linear(input_size,hidden_size)
                  self.fc2 = nn.Linear(hidden_size,output_size)
                  self.dropout = nn.Dropout(p=dropout_p)
        def forward(self,x) :
                  print("============== FeedForward NN  ===============", x.size())
                  x = F.relu(self.fc1(x))
                  x = self.dropout(x)
                  x = self.fc2(x)
                  return x
        
class EncoderLayer(nn.Module) :
     def __init__(self,config) :
        super(EncoderLayer,self).__init__()
        self.model_d = config.model_d
        self.max_length = config.max_sequence_len
        self.hidden_size = config.hidden_size
        self.attention_heads = config.n_heads
        self.dropout1 = nn.Dropout(p=config.dropout_p)
        self.dropout2 = nn.Dropout(p=config.dropout_p)
        self.multi_head_attention =  MultiHeadAttention(config)
        self.layernorm = LayerNorm()
        self.fc = FeedForward(self.model_d,self.model_d,self.hidden_size)

     def forward(self,x,mask=None) :
              res_x = x.clone()
              x = self.multi_head_attention(x,mask)
              x = self.dropout1(x)
              x = self.layernorm(x + res_x )
              res_x = x.clone()
              x = self.fc(x)
              x = self.dropout2(x)
              x =  self.layernorm(x + res_x )
              return x


class EncoderLayers(nn.Sequential) :

          def forward(self, x,mask=None):
                     for module in self._modules.values():
                          x = module(x,mask)
                     return x
          


class Encoder(nn.Module):
           def __init__(self,config) :
                super(Encoder,self).__init__()
                self.layers  = EncoderLayers(*[ EncoderLayer(config) for _ in range(config.n_layers)])
                self.tokenizer = Tokenizer(config,"encoder")
           def forward(self,x,mask,is_starttoken,is_endtoken) :
                   x = self.tokenizer(x,is_starttoken,is_endtoken)
                   x = self.layers(x,mask)
                   return x


class DecoderLayer(nn.Module) :
        def __init__(self,config) :
            super(DecoderLayer,self).__init__()
            self.model_d = config.model_d
            self.max_sequence_length = config.max_sequence_len
            self.hidden_size = config.hidden_size
            self.attention_heads = config.n_heads
            self.dropout1 = nn.Dropout(p=config.dropout_p)
            self.dropout2 = nn.Dropout(p=config.dropout_p)
            self.dropout3 = nn.Dropout(p=config.dropout_p)
            self.multi_head_attention =  MultiHeadAttention(config)
            self.layernorm = LayerNorm()
            self.fc = FeedForward(self.model_d,self.model_d,self.hidden_size)
        def forward(self,x,encoder_out,att_mask,pad_mask) :
                    res_x = x.clone()
                    x = self.multi_head_attention(x,att_mask)
                    x = self.dropout1(x)
                    x = self.layernorm(x + res_x )
                    res_x = x.clone()
                    x  = self.multi_head_attention((x,encoder_out,encoder_out),pad_mask)
                    x = self.dropout2(x)
                    x = self.layernorm(x + res_x )
                    res_x = x.clone()
                    x =  self.fc(x)
                    x = self.dropout3(x)
                    x = self.layernorm(x + res_x )
                    return x

class DecoderLayers(nn.Sequential) :

          def forward(self, x, encoder_out,att_mask,pad_mask):
                     for module in self._modules.values():
                          x = module(x,encoder_out,att_mask,pad_mask)
                     return x
          


class Decoder(nn.Module):
           def __init__(self,config) :
                super(Decoder,self).__init__()
                self.layers  = DecoderLayers(*[ DecoderLayer(config) for _ in range(config.n_layers)])
                self.tokenizer = Tokenizer(config,"decoder")

           def forward(self,x,encoder_out,att_mask,pad_mask,is_starttoken,is_endtoken) :
                   x = self.tokenizer(x,is_starttoken,is_endtoken)
                   x = self.layers(x,encoder_out,att_mask,pad_mask)
                   return  x

class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.model_d, config.ger_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                x,
                y,
                encoder_pad_mask=None,
                decoder_att_mask=None,
                decoder_pad_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x,encoder_pad_mask ,is_starttoken=enc_start_token, is_endtoken=enc_end_token)
        out = self.decoder(y,x,decoder_att_mask,decoder_pad_mask, is_starttoken=dec_start_token, is_endtoken=dec_end_token)
        out = self.linear(out)
        return out
    
