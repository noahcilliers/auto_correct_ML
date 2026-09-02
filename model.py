import torch
import numpy as np
import sys
import random
from torch import nn
import torch.nn.functional as F
from spellcheck.py import FREQUENCIES


# ok so how are we going to transform this long list into an actual voacabullary???
#
# we'll start with a simple NN that doesnt use context. this will make an inital implementation
# relatively simple. given a word list of this long. This is what we want to find

CHAR_EMB_DIM= 16
WORD_EMB_DIM = 20 * CHAR_EMB_DIM

#
# OK lets get the data in now
#
pairs = [(w, c) for w, c in FREQUENCIES.items() if ok.match(w) and 2 <= len(w) <= 20]
words = [w for w, _ in pairs]                          # list of ~83k str
weights = np.array([c for _, c in pairs], dtype=np.float64) ** 0.4
weights /= weights.sum()

rng = np.random.default_rng(0)
idx = rng.permutation(len(words))
cut = int(len(words) * 0.95)

# list of indeces which are the data we have for training and testing
train_idx, val_idx = idx[:cut], idx[cut:]

#so we can use a lstm to encode the meaning of the input sequence into some hidden units
# and then decoder will take that sequence and output it
## Single LSTM cell logic. we can send some data in and get this data out
class LSTMCell(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        combined_dim = hidden_dim+embedding_dim
        ### GATES
        self.forget_gate = nn.Linear(combined_dim, hidden_dim)
        self.input_gate = nn.Linear(combined_dim, hidden_dim)
        self.candidate_gate = nn.Linear(combined_dim, hidden_dim)
        self.output_gate = nn.Linear(combined_dim, hidden_dim)

        # applied to the non-recurrent connections only: embedding -> cell,
        # and cell -> logits. never to the h we hand to the next timestep.
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x, h, c):
        """
        So the cell state doesnt necassarily pass through a gate:
        it just is edited by these matmuls and then 
        used to create the hidden dim. So i guess it goes through some matrices
        but the matrices it goes through are different every time
        """


        embeds = self.dropout(x)
        combined = torch.cat([embeds, h], dim=1)
        # Gate functions below
        # These operations get all the values from our gates
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        candidate = torch.tanh(self.candidate_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        # These are the operations we provide for the cell state and the hidden 
        # forget(c * f), add info(i * x), new hidden with memory (o * tanh(c))
        c = c * f + i * candidate
        h = o * torch.tanh(c)


        return h, c


class StackedLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.cells = nn.ModuleList([
            LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    # dont we need c for each proir layer
    def forward(self, x, state=None, mask=None):
        # shape of the input
        B, T, _ = x.shape
        
        if state is None:
            h = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]
            c = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]
        else:
            h, c = [list(s.unbind(0)) for s in state]

        outs = []
        for t in range(T):
            # now we can actually run the forward pass through the stacked lstm
            inp = x[:, t]
            m = None if mask is None else mask[:, t].unsqueeze(1)
            for l, cell in enumerate(self.cells):
                h_new, c_new = cell(inp, h[l], c[l])
                if m is not None:
                    h_new = torch.where(m, h_new, h[l])
                    c_new = torch.where(m, c_new, c[l])
                h[l], c[l] = h_new, c_new
                inp = h[l]
            outs.append(inp)

        return torch.stack(outs, 1), (torch.stack(h), torch.stack(c))




class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, emb, num_layers=4):
        super().__init__()
        self.lstm = StackedLSTM(embedding_dim, hidden_dim, num_layers)
        self.emb = emb

    def forward(self, x):
        _, state = self.lstm(self.emb(x))
        return state



class Decoder(nn.Module):  
    def __init__(self, embedding_dim, vocab_size, hidden_dim, emb, num_layers=4):
        super().__init__()
        self.emb = emb
        self.lstm = StackedLSTM(embedding_dim, hidden_dim, num_layers)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.out.weight = emb.weight

    def forward(self, x, state):
        outs, state = self.lstm(self.emb(x), state)
        logits = self.out(outs)
        return logits, state





class Seq2Seq(nn.Module):   # owns the shared embedding, wires enc -> dec
    def __init__(self, vocab_size, embedding_dim, hidden_dim, emb):
        super().__init__()
        emb = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim, emb, num_layers=4)
        self.decoder = Decoder(embedding_dim, vocab_size, hidden_dim, emb, num_layers=4)

    # can we input the entire sequence into the forward pass
    # this forward pass is for training only. For generation we need to make a different 
    # function. Since the previous input must be used instead of an array
    def forward(self, src, tgt_in):
        # run the loop for encoder
        state = self.encoder(src)
        # run the loop for decoder
        logits, _ = self.decoder(tgt_in, state)
        return logits


    # generate one sequence
    @torch.no_grad()
    def generate(self, src, max_len=100):
        # get the state with the encoder
        state = self.encoder(src)
        tok = torch.full((src.size(0), 1), SOS, device=src.device)
        out = []
        # run until decoder says <eos>
        for _ in range(max_len):
            logits, state = self.decoder(tok, state)      # [B, 1, V]
            tok = logits[:, -1].argmax(-1, keepdim=True)  # [B, 1]
            if tok.item() == EOS: break
            out.append(tok)
            #output the tensor of outputs 
        return torch.cat(out, 1)


