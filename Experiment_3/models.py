import torch
import torch.nn.functional as F

from early_stop_solver import EarlyStopInt
from base_layers import BaseGNN
from layers import LaplacianODEFunc, ConstantODEblock

class GRAND(BaseGNN):
    def __init__(self, opt, w_config, dataset, device=torch.device('cpu')):
        super(GRAND, self).__init__(opt, w_config, dataset, device)
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        
        self.device = device
        time_tensor = torch.tensor([0, self.T]).to(device)  
        
        self.odeblock = block(self.f, self.regularization_fns, opt, w_config, dataset.data, device, t=time_tensor).to(device)
        
        with torch.no_grad():
            self.odeblock.test_integrator = EarlyStopInt(self.T, self.opt, self.device)
            self.set_solver_data(dataset.data)
    
    def set_solver_m2(self):
        self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
        self.odeblock.test_integrator.m2_bias = self.m2.bias.data.detach().clone().to(self.device)


    def set_solver_data(self, data):
        self.odeblock.test_integrator.data = data

    def forward(self, x, w_config, pos_encoding=None): 
        
        # Encode each node based on its feature.
        if self.opt['use_labels']:
            y = x[:, -self.num_classes:]
            x = x[:, :-self.num_classes]

        ## encoding
        if self.opt['beltrami']:
            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.mx(x)
            p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
            p = self.mp(p)
            x = torch.cat([x, p], dim=1)
        else:
            x = F.dropout(x, self.opt['input_dropout'], training=self.training) 
            x = self.m1(x) 
        
        self.odeblock.set_x0(x)

        with torch.no_grad(): 
            self.set_solver_m2()
    
        if self.training  and self.odeblock.nreg > 0: 
            z, self.reg_states  = self.odeblock(x, w_config)
        else:
            z = self.odeblock(x, w_config)
        
        z = F.relu(z)

        z = F.dropout(z, self.opt['dropout'], training=self.training) 

        # Decoding
        z = self.m2(z) 
    
        return z