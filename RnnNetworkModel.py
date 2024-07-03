import common_resources
torch,device=common_resources.get_torch_data()

class NETWORK(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.device=device
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
     
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim,batch_first=True)   
        self.rnn1 = torch.nn.LSTM(hidden_dim,
                                 300,batch_first=True) 
        
        self.rnn2 = torch.nn.LSTM(300,
                                 200,batch_first=True) 
             
        self.rnn3 = torch.nn.LSTM(200,
                                 hidden_dim,batch_first=True) 
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        self.batch_norm =torch.nn.BatchNorm1d(hidden_dim)

        #initialize hidden state
        self.hidden_states = [None, None, None, None]

        

    def forward(self, text,text_length):
        # if torch.any(text >= self.embedding.num_embeddings) or torch.any(text < 0):
        #     raise ValueError("Embedding indices are out of range")
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'), batch_first=True,enforce_sorted=False)
        
        packed_output, self.hidden_states[0] = self.rnn(packed,self.hidden_states[0])

        packed_output1, self.hidden_states[1] = self.rnn1(packed_output,self.hidden_states[1])
        packed_output2, self.hidden_states[2] = self.rnn2(packed_output1,self.hidden_states[2])
        packed_output3, self.hidden_states[3] = self.rnn3(packed_output2,self.hidden_states[3])
        h_n=self.hidden_states[3][0]
        last_hidden_state = h_n[-1]
        last_hidden_state = self.batch_norm(last_hidden_state)
        output = self.fc(last_hidden_state)
        return output
    
    def reset_hidden_state(self):
        self.hidden_states = [None, None, None, None]

    def _process_hidden_state(self,data):
        if data is not None:
            return (data[0].detach(),data[1].detach())
        

    def save_hidden_state(self):
        # print(self.hidden_states)
        return [ self._process_hidden_state(i) for i in self.hidden_states ]

    def load_hidden_state(self, hidden_states):
        self.hidden_states = hidden_states