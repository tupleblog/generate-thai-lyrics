import torch.nn as nn

train_on_gpu = False
class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                      embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                            hidden_size=self.hidden_dim, 
                            dropout=self.dropout,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
    
    
    def forward(self, nn_input, hidden):
        batch_size, _ = nn_input.size() # batch first
        embedding_input = self.embedding(nn_input)
        nn_output, hidden = self.lstm(embedding_input, hidden)
        nn_output = nn_output.contiguous().view(-1, self.hidden_dim)
        
        output = self.fc(nn_output)
        output = output.view(batch_size, -1, self.output_size)
        output = output[:, -1]

        # return one batch of output word scores and the hidden state
        return output, hidden
    
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        # initialize hidden state with zero weights, and move to GPU if available
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden