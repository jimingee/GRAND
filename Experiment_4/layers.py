import torch
from torch import nn
import torch_sparse

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from base_layers import ODEblock, ODEFunc


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
    
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                device=edge_index.device)
        

    if not fill_value == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes) # 노드 self-loop 추가
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    
    return edge_index, edge_weight


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, regularization_fns, opt, w_config, data, device, t=torch.tensor([0, 1])):
        super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt,w_config, data, device, t)

        self.aug_dim = 2 if opt['augment'] else 1 

        self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, w_config, data, device)
        if opt['data_norm'] == 'rw': 
            edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                    fill_value=opt['self_loop_weight'],
                                                                    num_nodes=data.num_nodes,
                                                                    dtype=data.x.dtype)
        # edge_index: [2, 12623]  / edge_weight: [12623]

        self.odefunc.edge_index = edge_index.to(device)
        self.odefunc.edge_weight = edge_weight.to(device)
        self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

        if opt['adjoint']: 
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.set_tol()

    def forward(self, x):
        t = self.t.type_as(x)

        integrator = self.train_integrator if self.training else self.test_integrator
        
        reg_states = tuple( torch.zeros(x.size(0)).to(x) for _ in range(self.nreg) )
        func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
        state = (x,) + reg_states if self.training and self.nreg > 0 else x
        
        if self.opt["adjoint"] and self.training:
        ## odeint -> differential 수행        
            state_dt = integrator(
                func, state, t,
                method=self.opt['method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.opt['tol_scale'],
                rtol=self.opt['tol_scale'],
                adjoint_atol=self.opt['tol_scale_adjoint'],
                adjoint_rtol=self.opt['tol_scale_adjoint'])
        else:
            state_dt = integrator(
                func, state, t,
                method=self.opt['method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol)
        
        if self.training and self.nreg > 0:
            z = state_dt[0][1]
            reg_states = tuple( st[1] for st in state_dt[1:] )
            
            
        
            return z, reg_states
        
        else: 
            z = state_dt[1]
        
        return z
        
    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
            + ")"


class MaxNFEException(Exception): pass

class LaplacianODEFunc(ODEFunc):

    def __init__(self, in_features, out_features, opt, w_config, data, device):
        super(LaplacianODEFunc, self).__init__(opt, w_config, data, device)

        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.eye(opt['hidden_dim'])) 
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def sparse_multiply(self, x):
        if self.opt['block'] in ['attention']:  
            mean_attention = self.attention_weights.mean(dim=1)
            ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
        elif self.opt['block'] in ['mixed', 'hard_attention']:  
            ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
        else: 
            ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
        return ax


    def forward(self, t, x): 
        if self.nfe > self.opt["max_nfe"]:
            print('Count of forward exceeded The MAX NFE')
            raise MaxNFEException
        self.nfe += 1
        ax = self.sparse_multiply(x)
        if not self.opt['no_alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train

        f = alpha * (ax - x)
        if self.opt['add_source']:
            f = f + self.beta_train * self.x0
        return f