import argparse
import numpy as np
import torch
import time
import wandb

from data import load_data, set_train_val_test_split
from models import GRAND


def print_model_params(model):
    print(model)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)

def train(model, w_config, optimizer, data, pos_encoding=None):
    model.train()
    
    optimizer.zero_grad()
    feat = data.x
    out = model(feat, w_config, pos_encoding)
    
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    
    if model.odeblock.nreg > 0:
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        regularization_coeffs = model.regularization_coeffs
        
        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )
        loss = loss + reg_loss
        
    model.fm.update(model.getNFE())
    model.resetNFE()
    
    loss.backward() 
    optimizer.step() 
    
    model.bm.update(model.getNFE()) 
    model.resetNFE()
    return loss.item() 



@torch.no_grad()
def test(model, w_config, data, pos_encoding=None, opt=None):  
    model.eval()
    feat = data.x
    logits, accs = model(feat, w_config, pos_encoding), []
    
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs


def build_optimizer(parameters, optimizer, learning_rate, weight_decay):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr = learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr = learning_rate)
    else:
        optimizer = torch.optim.Adamax(parameters, lr= learning_rate, weight_decay=weight_decay)
    
    return optimizer

def main(opt, w_config):
    dataset = load_data(opt, opt['not_lcc'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_encoding = None
    
    model = GRAND(opt, w_config, dataset,device).to(device)
    
    dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data, num_development=1500)
    data = dataset.data.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = build_optimizer(parameters, opt['optimizer'], learning_rate=opt['learning_rate'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0
    
    for epoch in range(opt['epoch']): 
        wandb.watch(model, log='all', log_freq = 10)
        start_time = time.time()
        
        loss = train(model, w_config, optimizer, data, pos_encoding)
        
        tmp_train_acc, tmp_val_acc, tmp_test_acc = test(model, w_config, data, pos_encoding, opt)
        
        best_time = opt['time']
        
        if tmp_val_acc > val_acc:
            best_epoch = epoch
            train_acc,val_acc,test_acc= tmp_train_acc, tmp_val_acc, tmp_test_acc
            best_time = opt['time']

        if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc: 
            best_epoch = epoch
            val_acc = model.odeblock.test_integrator.solver.best_val
            test_acc = model.odeblock.test_integrator.solver.best_test
            train_acc = model.odeblock.test_integrator.solver.best_train
            best_time = model.odeblock.test_integrator.solver.best_time
            
        wandb.log({'loss_train':loss, 'acc_train':tmp_train_acc, 'acc_val':tmp_val_acc, 'acc_test':tmp_test_acc, 'time':time.time() - start_time}, step=epoch)
        
        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'
        print(log.format(epoch+1, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
        
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc, best_epoch, best_time))
    return train_acc, val_acc, test_acc



def set_config():
    # CUDA_VISIBLE_DEVICES=2
    
    parser = argparse.ArgumentParser()
    
    # data args
    parser.add_argument('--dataset', default='Cora', type=str,
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', default = False, dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--geom_gcn_splits', default = False, dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from ')
    
    
    # rewiring args
    parser.add_argument("--not_lcc", default= True, action="store_false", help="don't use the largest connected component")
    
    

    # GNN args
    parser.add_argument('--fc_out', default=False, dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument("--batch_norm", default=False, dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--decay', type=float, default=0.00507685443154266, help='Weight decay for optimization')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--use_mlp', default=False, dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', default=True, dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--no_alpha_sigmoid', default=False, dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--hidden_dim', type=int, default=80, help='Hidden dimension.') 
    parser.add_argument('--dropout', type=float, default=0.04, help='Dropout rate.')
    parser.add_argument('--learning_rate', type=float, default=0.0229, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--optimizer', type=str, default='adamax', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    
    

    # ODE args
    parser.add_argument('--time', type=float, default=18.294754260552843, help='End time of ODE integrator.')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--adjoint', default=True, dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument("--adjoint_method", default= 'euler', type=str, 
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--method', default = 'rk4',type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint") # 이것도 euler로 수정 필요해 보임
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--max_nfe", type=int, default=2000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                            "used if getting OOM errors at test time")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--no_early", action="store_true",default=False,
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    ## 튜닝할 ODE 기반 하이퍼파라미터 
    #parser.add_argument('--step_size', type=float, default=1,
    #                    help='fixed step size when using fixed step solvers e.g. rk4')
    #parser.add_argument('--adjoint_step_size', type=float, default=1,
    #                    help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    #parser.add_argument('--tol_scale', default = 821.9773048827274, type=float, help='multiplier for atol and rtol')
    #parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
    #                    help="multiplier for adjoint_atol and adjoint_rtol")
    

    ## beltrami
    parser.add_argument('--beltrami', default = False, action='store_true', help='perform diffusion beltrami style')


    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    args = parser.parse_args()
    opt = vars(args)
    return opt

    ## sweep 할 parameter 설정
sweep_config = {
    'name' : 'sweep-GRAND-args',
    'method' : 'random',
    'metric' : {'name' : 'acc_val','goal' : 'maximize'},
    'parameters':{
        ## sweep 대상
        'step_size':{'min':0.1, 'max': 2.0}, # default 1
        'adjoint_step_size' : {'min': 0.1, 'max':2.0}, # default 1
        'tol_scale' : {'min':100.0, 'max':1000.0}, # default : 821.9773048827274
        'tol_scale_adjoint':{'min':0.1, 'max':2.0} # default 1.0  
    }
    
}


cnt = 1
def run_sweep(config = None):
    global cnt
    opt = set_config() # 조정이 필요하지 않은 하이퍼 파라미터   
    wandb.init(config=config)
    wandb.run.name = f'GRAND_args_{cnt}'
    w_config = wandb.config
    cnt += 1

    main(opt, w_config)
    
sweep_id = wandb.sweep(sweep=sweep_config, project = 'sweep-grand-params', entity='jimin-choi')
wandb.agent(sweep_id, function = run_sweep, count= 100)