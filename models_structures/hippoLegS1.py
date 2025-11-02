import torch 
import torch.nn as nn 



def matrix(hippo_type , dim) :
    if hippo_type =='legs' : 
        A = torch.zeros(dim , dim)
        b  = torch.zeros(dim , 1)
        for i in range(dim) : 
            A[i , i] = i+1 
            b[i] = (2*(i+1))**(1/2)
        for i in range(dim) : 
            for j in range(0 , i , 1) : 
                A[i , j] = ((i+1)*(2*j+1))**0.5
        return A , b
    elif hippo_type == 'random' : 
        A = torch.randn(dim , dim)
        b = torch.randn(dim , 1)
        
def discretisization(A , b  , n_samples) : 
    dim = A.shape[0]
    I = torch.eye(dim)
    A_d = [] 
    b_d = []
    for k in range(n_samples) : 
        A_d.append(I - A/(k+1))
        b_d.append(b/(k+1))
    return torch.stack(A_d , dim=0) , torch.stack(b_d , dim=0)


class RNN_block(nn.Module) : 
    def __init__(self, x_dim, h_dim, c_dim , y_dim) : 
        super().__init__()
        self.Wxh = nn.Parameter(torch.randn(x_dim+c_dim ,h_dim  ) ,requires_grad=True)
        self.Whh = nn.Parameter(torch.randn(h_dim , h_dim) , requires_grad=True)
        self.bh = nn.Parameter(torch.randn(1 , h_dim ) , requires_grad=True)
        self.Wyh = nn.Parameter(torch.randn( h_dim , y_dim ) , requires_grad=True)
        self.by = nn.Parameter(torch.randn(1 , y_dim))
        self.tanh = nn.Tanh()
        self.f = nn.Linear(h_dim , 1)
    def forward(self , x  , h_previous  , c_previous , A , b) : 
        #x : (x_dim  , 1 )
        xc  = torch.concat([x , c_previous] , dim=1)
        h_next = self.tanh(xc @self.Wxh + h_previous@self.Whh   + self.bh)
        y =h_next  @self.Wyh  + self.by
        c_next = (A @ c_previous.T).T +  self.f(h_next) @ b.T
        return h_next ,c_next ,  y









class RNN(nn.Module):
    def __init__(self, x_dim, h_dim, dim_c , y_dim , len_sequence):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnnCell = RNN_block(x_dim, h_dim, dim_c , y_dim)
        self.h_dim = h_dim
        self.c_dim = dim_c
        self.len_sequence = len_sequence
        A , b = matrix('legs' , dim_c)
        Ad ,bd = discretisization( A , b , len_sequence)
        self.register_buffer("Ad", Ad)
        self.register_buffer("bd", bd)
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.shape[0]
        device = x.device
        h_next = torch.zeros(batch_size, self.h_dim, device=device)
        c_next = torch.zeros(batch_size, self.c_dim, device=device)
        out = []
        for i in range(self.len_sequence):
            x_t = x[:, i, :] # shape (batch, features )
            h_next, c_next , y = self.rnnCell(x_t, h_next ,c_next, self.Ad[i , : , : ] , self.bd[i , :])
            out.append(y)  # append (batch, y_dim)
        out = torch.stack(out, dim=1)  # (batch, seq_len, y_dim)
        return out




        
class model(nn.Module) : 
    def __init__(self , x_dim, h_dim, c_dim  , seq_len , dim2 , dim3 , dim_out) : 
        super().__init__()
        self.hippo_rnn =RNN(x_dim, h_dim, c_dim , 1 , seq_len)
        self.fc1 = nn.Linear(seq_len , dim2)
        self.fc2 = nn.Linear(dim2 ,dim3)
        self.fc3 = nn.Linear(dim3 ,  dim_out)
        self.relu = nn.ReLU()
    def forward(self , x ) :
        x = self.hippo_rnn(x).squeeze(-1)
        x = self.relu(x)
        x= self.fc1(x)
        x = self.relu(x)
        x= self.fc2(x)        
        x = self.relu(x)
        x= self.fc3(x)
        return x
