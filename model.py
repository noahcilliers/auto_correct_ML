import torch
import sys
import random
from torch import nn
import torch.nn.functional as F
from data import synthetic, norvig, encode, decode, PAD, SOS, EOS, VOCAB_SIZE


# ok so how are we going to transform this long list into an actual voacabullary???
#
# we'll start with a simple NN that doesnt use context. this will make an inital implementation
# relatively simple. given a word list of this long. This is what we want to find

CHAR_EMB_DIM= 16

#
# the data is two lists of (wrong, right) pairs from data.py: synthetic() for
# corrupted en_words and norvig() for the real (train, val) split. batching
# lives below, next to the training code.
#

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
    

    def forward(self, x, state):
        outs, state = self.lstm(self.emb(x), state)
        logits = self.out(outs)
        return logits, state





class Seq2Seq(nn.Module):   # owns the shared embedding, wires enc -> dec
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
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



CHECKPOINT = ""


# ------------------------------------------------------------------- batching
#
# The lists from data.py are plain (wrong, right) pairs. These turn them into
# batches, and to_tensors() into padded tensors. Padding is per batch, to the
# longest word in it, so there is no fixed seq_len.

def make_batches(pairs, size, seed=None):
    """One shuffled pass over `pairs`, yielding lists of `size` (the last may be short)."""
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        yield [pairs[j] for j in order[i:i + size]]


def grab(pairs, size, rng=random):
    """A single random batch of `size` distinct pairs."""
    return rng.sample(pairs, size)


def mixed_batches(synth, norvig_train, size, norvig_frac=0.3, seed=None):
    """
    Phase-2 batches: one shuffled pass over `synth`, each batch topped up to
    `size` with `norvig_frac` of Norvig pairs drawn at random (with replacement).
    """
    rng = random.Random(seed)
    n_norvig = round(size * norvig_frac)
    for chunk in batches(synth, size - n_norvig, seed=rng.random()):
        batch = chunk + rng.choices(norvig_train, k=n_norvig)
        rng.shuffle(batch)
        yield batch


def to_tensors(batch, device):
    """
    A list of (wrong, right) pairs -> (src, tgt_in, tgt_out) long tensors on
    `device`, each [B, T] padded with PAD. Mask with `!= PAD`, and give the
    loss ignore_index=PAD so padded positions of tgt_out don't count.
    """
    src, tgt_in, tgt_out = encode(batch)
    return (torch.tensor(src, device=device),
            torch.tensor(tgt_in, device=device),
            torch.tensor(tgt_out, device=device))


# ------------------------------------------------------------------- training

def __train__(model, device, synth, norvig_train, norvig_val, optimizer, epochs, batch_size, cont):
    # phase 1:  for batch in batches(synth, batch_size): ...
    model.train()
    for e in range(epochs):
        batches = make_batches(synth, batch_size)
        for batch in batches:
            #run through the network
            src, tgt_in, tgt_out = to_tensors(batch, device)
            optimizer.zero_grad()
            # preds
            logits = model(src, tgt_in)
            # loss
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1), ignore_index=PAD)
            # backprop
            loss.backward()
            #apply grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # back prop

    # phase 2:  for batch in mixed_batches(synth, norvig_train, batch_size): ...
    # validate: for batch in batches(norvig_val, batch_size, seed=0): ...
    # each batch goes through to_tensors(batch, device) first.
    return 0



def train(cont, vocab_size=VOCAB_SIZE, embedding_dim=CHAR_EMB_DIM, hidden_dim=256):
    # this function will create or grab the model and then set all the proper parameters
    # for the training
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Seq2Seq(vocab_size, embedding_dim, hidden_dim).to(device)

    if cont:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    batch_size = 64

    # the datasets: lists of (wrong, right) pairs
    synth = synthetic(500_000)            # phase 1, and 70% of each phase-2 batch
    norvig_train, norvig_val = norvig()   # the 30% mix-in, and what we validate on
    
    __train__(model, device, synth, norvig_train, norvig_val, optimizer, epochs, batch_size, cont)


def main():
    synth = synthetic(100)
    print(synth)


if __name__ == "__main__":
    main()
