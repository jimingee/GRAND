## Regularized ODEfunc
## ODEblock
## ODEFunc

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from utils import Meter
import six


def quadratic_cost(x, t, dx, unused_context):
    del x, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).mean(dim=-1)

def divergence_bf(dx, x):
    sum_diag = 0.
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def jacobian_frobenius_regularization_fn(x, t, dx, context):
    del t
    return divergence_bf(dx, x)


def total_derivative(x, t, dx, unused_context):
    del unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1 / x.numel(), requires_grad=True)
        tmp = torch.autograd.grad((u * dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError(
                'No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead')

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5 * tdv2.mean(dim=-1)

def directional_derivative(x, t, dx, unused_context):
    del t, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0), -1)

    return 0.5 * ddx2.mean(dim=-1)


class RegularizedODEfunc(nn.Module):
  def __init__(self, odefunc, regularization_fns):
    super(RegularizedODEfunc, self).__init__()
    self.odefunc = odefunc
    self.regularization_fns = regularization_fns

  def before_odeint(self, *args, **kwargs):
    self.odefunc.before_odeint(*args, **kwargs)

  def forward(self, t, state):
    with torch.enable_grad():
      x = state[0]
      x.requires_grad_(True)
      t.requires_grad_(True)
      dstate = self.odefunc(t, x)
      if len(state) > 1:
        dx = dstate
        reg_states = tuple(reg_fn(x, t, dx, self.odefunc) for reg_fn in self.regularization_fns)
        return (dstate,) + reg_states
      else:
        return dstate

  @property
  def _num_evals(self):
    return self.odefunc._num_evals



REGULARIZATION_FNS = {
    "kinetic_energy": quadratic_cost,
    "jacobian_norm2": jacobian_frobenius_regularization_fn,
    "total_deriv": total_derivative,
    "directional_penalty": directional_derivative
}

# regularization (규제 함수, 규제 계수 설정)
def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs



class ODEblock(nn.Module):
    def __init__(self, odefunc, regularization_fns, opt, w_config, data, device, t):
        super(ODEblock, self).__init__()
        self.opt = opt
        self.t = t
    
        self.aug_dim = 2 if opt['augment'] else 1 
        self.odefunc = odefunc(self.aug_dim * w_config.hidden_dim, w_config.hidden_dim, opt,w_config,  data, device)

        self.nreg = len(regularization_fns)
        self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)
        
        if opt['adjoint']: 
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
            
        self.train_integrator = odeint
        self.test_integrator = None
        self.set_tol()
        
    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()
        self.reg_odefunc.odefunc.x0 = x0.clone().detach()
        
    def set_tol(self):
        ## rtol: Relative tolerance / atol: Absolute tolerance
        self.atol = self.opt['tol_scale'] * 1e-7 
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']: 
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9
            
    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9
        
    def set_time(self, time):
        self.t = torch.tensor([0, time]).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) + ")"

class ODEFunc(MessagePassing):
    def __init__(self, opt, w_config, data, device):
        super(ODEFunc, self).__init__()
        self.opt = opt
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.x0 = None
        self.nfe = 0
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__


class BaseGNN(MessagePassing):
    def __init__(self, opt, w_config,dataset, device=torch.device('cpu')):
        super(BaseGNN, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.num_classes = dataset.num_classes
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.fm = Meter()
        self.bm = Meter()
        
        if opt['beltrami']: 
            self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
            self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
            w_config.hidden_dim = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
        else:
            self.m1 = nn.Linear(self.num_features, w_config.hidden_dim) 

        if self.opt['use_mlp']: 
            self.m11 = nn.Linear(w_config.hidden_dim, w_config.hidden_dim)
            self.m12 = nn.Linear(w_config.hidden_dim, w_config.hidden_dim)
        if opt['use_labels']: 
            w_config.hidden_dim = w_config.hidden_dim + dataset.num_classes
        else:
            self.hidden_dim = w_config.hidden_dim
        if opt['fc_out']: 
            self.fc = nn.Linear(w_config.hidden_dim, w_config.hidden_dim)
            
        self.m2 = nn.Linear(w_config.hidden_dim, dataset.num_classes) 
        if self.opt['batch_norm']: 
            self.bn_in = torch.nn.BatchNorm1d(w_config.hidden_dim)
            self.bn_out = torch.nn.BatchNorm1d(w_config.hidden_dim)

        self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

    def getNFE(self):
        return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0
        self.odeblock.reg_odefunc.odefunc.nfe = 0

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__