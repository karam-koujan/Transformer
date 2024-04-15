from model import Transformer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = get_device()



file_path = "./deu.txt"
start_token = '<sof>'
end_token = '<eof>'
pad_token = '<pad>'
english_vocabulary = [start_token,'<unk>', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/','`','’',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@','[', '\\', ']', '^', '_', '`',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z',
                        '{', '|', '}', '~', pad_token, end_token]

german_vocabulary = [start_token,'<unk>',' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/','`','’',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', 'ä', 'ö', 'ü', 'ß',
                      '{', '|', '}', '~', pad_token, end_token]


index_to_german = {k : v  for k,v in enumerate(german_vocabulary)}
german_to_index = {v : k  for k,v in enumerate(german_vocabulary)}
index_to_english = {k : v  for k,v in enumerate(english_vocabulary)}
english_to_index = {v : k  for k,v in enumerate(english_vocabulary)}


with open(file_path, 'r') as file:
    raw_data = file.readlines()
sentences =  [ (sentence.rstrip("\n").split("\t")[0].lower(),sentence.rstrip("\n").split("\t")[1].lower()) for sentence in raw_data]

PERCENTILE = 97
print( f"{PERCENTILE}th percentile length English: {np.percentile([len(x[0]) for x in sentences], PERCENTILE)}" )
print( f"{PERCENTILE}th percentile length German: {np.percentile([len(x[1]) for x in sentences], PERCENTILE)}" )

def is_token_exist(sentence,vocab):
     for token in sentence :
           if token not in vocab :
                 return False
     return True

def  is_valid_length(sentence,max_sequence_length) :
           return len(sentence) < max_sequence_length - 1

is_token_exist('sie geht zu fuß.',german_vocabulary)

valid_sentence_indicies = []
for index in range(len(sentences)):
    german_sentence, english_sentence = sentences[index][1], sentences[index][0]
    if is_valid_length(german_sentence, max_sequence_len) \
      and is_valid_length(english_sentence, max_sequence_len) \
      and is_token_exist(german_sentence, german_vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


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


model_d = 512
batch_size = 30
n_heads = 8
n_layers = 1
max_sequence_len = 200
ger_vocab_size = len(german_vocabulary)
eng_vocab_size = len(english_vocabulary)
config = Config(model_d=model_d, max_sequence_len=max_sequence_len, ger_vocab_size=ger_vocab_size, eng_vocab_size=eng_vocab_size, n_layers=n_layers, n_heads=n_heads, eng_to_index=english_to_index, ger_to_index=german_to_index, start_token=start_token, end_token=end_token, pad_token=pad_token)
transformer = Transformer(config)

class TextDataset(Dataset) :
    def __init__(self,sentences) :
         self.sentences = sentences


    def __len__(self) :
         return len(self.sentences)
    def __getitem__(self,idx):
           german = self.sentences[idx][1]
           english = self.sentences[idx][0]
           return english,german


criterian = nn.CrossEntropyLoss(ignore_index=german_to_index[pad_token])
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)
optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)

def create_masks(eng_batch, ger_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.zeros((max_sequence_len, max_sequence_len))
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.zeros((num_sentences, max_sequence_len, max_sequence_len))
    decoder_padding_mask_self_attention = torch.zeros((num_sentences, max_sequence_len, max_sequence_len))
    decoder_padding_mask_cross_attention = torch.zeros((num_sentences, max_sequence_len, max_sequence_len))

    for idx in range(num_sentences):
      eng_sentence_length, ger_sentence_length = len(eng_batch[idx]), len(ger_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length  , max_sequence_len)
      ger_chars_to_padding_mask = np.arange(ger_sentence_length , max_sequence_len)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = 1
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = 1
      decoder_padding_mask_self_attention[idx, :, ger_chars_to_padding_mask] = 1
      decoder_padding_mask_self_attention[idx, ger_chars_to_padding_mask, :] = 1
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = 1
      decoder_padding_mask_cross_attention[idx, ger_chars_to_padding_mask, :] = 1
    decoder_self_attention_mask =   decoder_padding_mask_self_attention + look_ahead_mask
    return encoder_padding_mask, decoder_self_attention_mask, decoder_padding_mask_cross_attention,

dataset = TextDataset(sentences)
train_loader = DataLoader(dataset,batch_size)

transformer.train()
transformer.to(device)
num_epochs = 10
losses  = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    epoch_loss = 0.0

    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, ger_batch = batch

        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, ger_batch)
        optim.zero_grad()
        ger_predictions = transformer(eng_batch,
                                     ger_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.tokenizer.batch_tokenization(ger_batch, is_starttoken=False, is_endtoken=True)
        loss = criterian(
            ger_predictions.view(-1, ger_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)

        loss = loss.sum()
        epoch_loss+=loss.item()
        loss.backward()
        optim.step()


    losses.append(epoch_loss/len(iterator))



transformer.eval()

ger_sentence = ("",)
eng_sentence = ("hello",)
with torch.no_grad():
        for word_counter in range(max_sequence_len):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, ger_sentence)
                predictions = transformer(eng_sentence,
                                          ger_sentence,
                                          encoder_self_attention_mask.to(device),
                                          decoder_self_attention_mask.to(device),
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                print("____predictions shape___",predictions.shape)
                next_token_prob_logits = predictions[0][word_counter]
                print(next_token_prob_logits,next_token_prob_logits.shape)
                next_token_index = torch.argmax(next_token_prob_logits).item()
                next_token = index_to_german[next_token_index]
                print("next token",next_token)
                ger_sentence = (ger_sentence[0]+next_token,)
                if next_token == end_token:
                  break

        print(f"{eng_sentence} : {ger_sentence}")
