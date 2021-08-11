import sys
sys.path.append("../")
sys.path.append("../../")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ShapeAssembly import Program, hier_execute, make_hier_prog, ShapeAssembly
import sa_utils as utils
import infer_recon_metrics
import argparse
import ast
import random
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from argparse import Namespace
import pickle
from make_abs_data import fillProgram
from tqdm import tqdm
import infer_sem_valid as sv
import numpy
import json
from pc_encoder import PCEncoder

MAX_LINES = 30

device = torch.device("cuda")

sa = ShapeAssembly()

outpath = "model_output"
SEQ_LEN = 0
MAX_DEPTH = 4

I_LENGTH = 11 # cuboid index
SQ_LENGTH = 6 # face index
SYM_LENGTH = 3 # axis index

VERBOSE = False

closs = nn.BCEWithLogitsLoss(reduction='none')
celoss = nn.CrossEntropyLoss(reduction='none')

FUNC_PRED_FIELD = 'func_pred'
CPARAM_PRED_FIELD = 'cparam_pred'
DPARAM_PRED_FIELD = 'dparam_pred'

TRAIN_LOG_INFO = [
    ('Total loss', 'loss', 'batch_count'),
    ('Func loss', 'func', 'batch_count'),
    ('Float Param loss', 'f_prm', 'batch_count'),
    ('Disc Param loss', 'd_prm', 'batch_count'),
    ('Bool Param loss', 'b_prm', 'batch_count'),
    ('BBox Param Loss', 'bbox', 'batch_count'),
    ('Child loss', 'child', 'batch_count'),
    ('KL loss', 'kl', 'batch_count'),
    ('Func Correct %', 'func_corr', 'func_total'),
    ('Disc Correct %', 'd_corr', 'd_total'),
    ('Bool Correct %', 'b_corr', 'b_total'),
    ('Child Correct %', 'c_corr', 'c_total'),
    ('Float Mean Error', 'f_prm', 'f_norm'),
]

EVAL_LOG_INFO = [
    ('CD', 'cd', 'no_norm'), 
    ('Fscore-Def', 'fscore-def', 'no_norm'),
    ('Fscore-01', 'fscore-01', 'no_norm'), 
    ('Fscore-03', 'fscore-03', 'no_norm'), 
    ('Fscore-05', 'fscore-05', 'no_norm'),
    ('Rooted', 'rooted', 'no_norm'),
    ('Stable', 'stable', 'no_norm'), 
    ('Prog Creation %', 'prog_creation_perc', 'no_norm'),
    ('Missing Line Number %', 'miss_ln', 'num_progs'),
    ('Extra Line Number %', 'extra_ln', 'num_progs'), 
    ('Corr Child Number %', 'cn_corr', 'num_progs'),            
    ('Func Correct %', 'func_corr', 'func_total'),
    ('Disc Correct %', 'd_corr', 'd_total'),
    ('Bool Correct %', 'b_corr', 'b_total'),
    ('Child Correct %', 'child_corr', 'child_total'),
    ('Float Mean Error', 'f_prm', 'num_progs'),
    ('BBox Mean Error', 'bbox', 'num_progs')
]

def weighted_mae_loss(input, target, weight):
    return torch.sum(weight * (input - target).abs())

def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

def assign(var_name, value):
    return '{} = {}'.format(var_name, value)

# Multi-layer perceptron helper function
class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x),.2)
        x = F.leaky_relu(self.l2(x),.2)
        return self.l3(x)

# Unused, but can be used 
class simplePCEncoder(nn.Module):

    def __init__(self, feat_len):
        super(simplePCEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)
        self.mlp2mu = nn.Linear(feat_len, feat_len)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]

        return self.mlp2mu(net)


def getBatchEncData(batch):    
    pcs = torch.stack(batch,dim=0).to(device)
    return pcs
    
def get_pc_encodings(batch, encoder):
    pcs = getBatchEncData(batch)
    
    codes = encoder(pcs)
        
    return codes
    
def getShapeEvalResult(pred_node, gt_node):
    # Corr line %, Func Corr %, Child Corr %, Bool Corr %, Disc Corr %, Float Mean Distance,
    # child num corr, line numc corr
    results = {}

    ln = min(pred_node[FUNC_PRED_FIELD].shape[0], gt_node['e_func_gt'][1:-1].shape[0])

    if pred_node[FUNC_PRED_FIELD].shape[0] < gt_node['e_func_gt'][1:-1].shape[0]:
        results['miss_ln'] = 1. 
    else:
        results['miss_ln'] = 0.

    if pred_node[FUNC_PRED_FIELD].shape[0] > gt_node['e_func_gt'][1:-1].shape[0]:
        results['extra_ln'] = 1. 
    else:
        results['extra_ln'] = 0.
    
    results['num_progs'] = 1.
    
    func_pred = pred_node[FUNC_PRED_FIELD][:ln].cpu()
    cparam_pred = pred_node[CPARAM_PRED_FIELD][:ln].cpu()
    dparam_pred = pred_node[DPARAM_PRED_FIELD][:ln].cpu()

    func_target = gt_node['e_func_gt'][1:1+ln].cpu()
        
    float_target = gt_node['e_float_target'][1:ln+1].cpu()
    float_mask = gt_node['e_float_mask'][1:ln+1].cpu()
    disc_target = gt_node['e_disc_target'][1:ln+1].cpu()
    disc_mask = gt_node['e_disc_mask'][1:ln+1].cpu()
    bool_target = gt_node['e_bool_target'][1:ln+1].cpu()
    bool_mask = gt_node['e_bool_mask'][1:ln+1].cpu()
    
    results['func_corr'] = (
        func_pred == func_target
    ).float().sum().item()
    results['func_total'] = ln
        
    results['b_corr'] = (
        (cparam_pred == bool_target).float() * bool_mask
    ).sum().item()
    results['b_total'] = (bool_mask.sum() + 1e-8).item()
    
    results['d_corr'] = (
        (dparam_pred == disc_target).float() * disc_mask
    ).sum().item()
    results['d_total'] = (disc_mask.sum() + 1e-8).item()
    
    results['f_prm'] = ((
        ((cparam_pred - float_target).abs() * float_mask).sum()
    ) / (float_mask.sum() + 1e-8)).item()

    cn = min(len(pred_node['children']), len(gt_node['children']))

    if len(pred_node['children']) == len(gt_node['children']):
        results['cn_corr'] = 1. 
    else:
        results['cn_corr'] = 0.
        
    results['child_corr'] = 0.
    results['child_total'] = cn
        
    for pred_child, gt_child in zip(pred_node['children'][:cn], gt_node['children'][:cn]):
        if len(pred_child) == 0 and len(gt_child) == 0:
            results['child_corr'] += 1.
        elif len(pred_child) > 0 and len(gt_child) > 0:
            results['child_corr'] += 1.
            child_results = getShapeEvalResult(pred_child, gt_child)            
            for key in child_results:
                results[key] += child_results[key]
            
    return results

def getBatchDecData(progs):
    seq = torch.stack([p['seq'] for p in progs],dim=0)
    inp_seq = seq[:,:-1,:]
    tar_seq = seq[:,1:,:]
    seq_weight = torch.stack([p['seq_mask'] for p in progs],dim=0)[:,1:]    

    fprm_weight = torch.stack([p['fprm_weight'] for p in progs], dim=0)[:, 1:]                    
    children = [p['children'] for p in progs]

    child_target = torch.stack([p['child_target'] for p in progs], dim=0)[:, 1:]
    child_weight = torch.stack([p['child_weight'] for p in progs], dim=0)[:, 1:]

    lnext_inds = ((child_target.bool()) & child_weight.bool()).nonzero().tolist()
    
    cnext_inds = []
    for i in range(len(progs)):
        for j in progs[i]["exp_inds"]:
            cnext_inds.append([i,j])

    return inp_seq, tar_seq, seq_weight, fprm_weight, children, \
        child_target, child_weight, lnext_inds, cnext_inds


# GRU recurrent Decoder
class dslDecoder(nn.Module):
    def __init__(self, hidden_dim, metadata):
        super(dslDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        self.input_dim = metadata['tl_size']
        self.bb_net = MLP(hidden_dim, hidden_dim, hidden_dim, 3)        
        self.inp_net = MLP(self.input_dim + MAX_DEPTH + 3, hidden_dim, hidden_dim, hidden_dim)
        
        self.max_cparams = metadata['max_cparams']
        self.num_funcs = len(metadata['cparam_map'])

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first = True)        
        
        self.max_children = metadata['max_children']         
        self.child_net = MLP(hidden_dim * 2, hidden_dim//2, hidden_dim//8, self.max_children)
        on = self.max_children * hidden_dim
        self.next_code_net = MLP(hidden_dim * 2, hidden_dim, hidden_dim, on)

        tl_map = metadata['tl_map']
        
        func_net = MLP(hidden_dim, hidden_dim//2, hidden_dim//4, self.num_funcs)
        start, end = tl_map['func']
        func_net.start = start
        func_net.end = end
        func_net.line_cond = None
        func_net.bb_cond = None
        func_net.name = 'func'        
        func_net.func = None
        func_net._type = 'func'
        net_list = [func_net]
                
        for _,func in metadata['func_map'].items():
            if len(metadata['dparam_map'][func]) > 0:
                low = 1e8
                high = -1e8
                for i, prm in enumerate(metadata['dparam_map'][func]):
                    start, end = tl_map[f'{func}_{prm}_{i}']
                    mlp = MLP(hidden_dim, hidden_dim //2, hidden_dim//4, end-start)
                    mlp.start = start
                    mlp.end = end
                    mlp.line_cond = None
                    mlp.bb_cond = False
                    mlp.func = func 
                    mlp.name = f'{func}_{prm}_{i}'
                    mlp._type = 'disc'
                    net_list.append(mlp)
                    low = min(low, start)
                    high = max(high, end)
            else:
                low = None
                high = None

            if f'{func}_f' in tl_map:
                start, end = tl_map[f'{func}_f']                                
                if low is None or high is None:                    
                    mlp = MLP(hidden_dim + 3, hidden_dim //2, hidden_dim//4, end-start)
                    mlp.line_cond = None
                else:
                    mlp = MLP(hidden_dim + 3 + high-low, hidden_dim //2, hidden_dim//4, end-start)
                    mlp.line_cond = (low, high)
                mlp.start = start
                mlp.end = end
                mlp.bb_cond = True
                mlp.func = func
                mlp.name = f'{func}_f'
                mlp._type = 'f'
                net_list.append(mlp)

            if f'{func}_b' in tl_map:
                start, end = tl_map[f'{func}_b']
                mlp = MLP(hidden_dim, hidden_dim //2, hidden_dim//4, end-start)
                mlp.start = start
                mlp.end = end
                mlp.line_cond = None
                mlp.bb_cond = False
                mlp.func = func
                mlp.name = f'{func}_b'
                mlp._type = 'b'
                net_list.append(mlp)

        self.net_list = nn.ModuleList(net_list)                        
    
    def train_forward(self, inp_seq, code, _bb_dims, _hier_ind, gt_seq):
        
        bb_dims = _bb_dims.unsqueeze(1).repeat(1,inp_seq.shape[1],1)
        
        hier_oh = torch.zeros(
            inp_seq.shape[0], inp_seq.shape[1], MAX_DEPTH, device=device
        )

        hier_oh[
            torch.arange(inp_seq.shape[0],device=device),
            :,
            _hier_ind
        ] = 1.0
                
        inp = self.inp_net(
            torch.cat(
                (inp_seq, bb_dims, hier_oh), dim=2)
        )
        
        gru_out, _ = self.gru(inp, code.unsqueeze(0).contiguous())
                
        fstart, fend = self.metadata['tl_map']['func']
        commands = torch.argmax(gt_seq[:,:,fstart:fend], dim = 2).flatten()

        flat_out = torch.zeros(commands.shape[0], inp_seq.shape[2], device=device).float()
        
        flat_gt_seq = gt_seq.reshape(commands.shape[0], -1)
        flat_bb_dims = bb_dims.reshape(commands.shape[0], -1)
        flat_gru_out = gru_out.reshape(commands.shape[0], -1)
        
        for net in self.net_list:                            
            if net.func is None:
                # Func net                
                flat_out[:,net.start:net.end] = net(flat_gru_out)
                
            else:                
                cmd_inds = (commands == net.func).nonzero().flatten()
                
                if net.line_cond is not None:
                    line_cond = flat_gt_seq[cmd_inds, net.line_cond[0]:net.line_cond[1]]
                else:
                    line_cond = torch.zeros(cmd_inds.shape[0], 0, device=device)
                                                    
                if net.bb_cond is True:
                    bb_cond = flat_bb_dims[cmd_inds,:]
                else:
                    bb_cond = torch.zeros(cmd_inds.shape[0], 0, device=device)
                                                            
                flat_out[cmd_inds, net.start:net.end] = net(torch.cat((
                    flat_gru_out[cmd_inds,:], line_cond, bb_cond
                ), dim=1))
                
        out = flat_out.view(inp_seq.shape)
        
        double_enc = torch.cat((
            gru_out, code.unsqueeze(1).repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
            
        child_pred = self.child_net(
            double_enc
        )
                
        next_codes = self.next_code_net(
            double_enc                        
        ).view(inp_seq.shape[0], inp_seq.shape[1], self.max_children, -1)                
       
        return out, next_codes, child_pred

    
    def calc_loss(self, out, pchild, tar, child_tar, seq_weight, fprm_weight, child_weight):
        result_map = {}
        
        result_map['f_prm'] = weighted_mae_loss(out, tar, fprm_weight)

        tl_map = self.metadata['tl_map']

        fstart, fend = tl_map['func']

        with torch.no_grad():
            commands = torch.argmax(tar[:,:,fstart:fend], dim = 2).flatten()
            pcommands = torch.argmax(out[:,:,fstart:fend], dim = 2).flatten()        
            result_map['func_corr'] = (
                (commands == pcommands).float() * seq_weight.flatten()
            ).sum().item() * 1.0
            result_map['func_total'] = seq_weight.sum().item()
            
        result_map['func'] = (celoss(
            out[:,:,fstart:fend].view(-1,fend-fstart),
            commands
        ) * seq_weight.flatten()).sum()

        result_map['child'] = (closs(pchild, child_tar) * child_weight).sum()

        result_map['c_corr'] = (((pchild >= 0).float() == child_tar).float() * child_weight).sum().item()
        result_map['c_total'] = (child_weight.sum() + 1e-8).item()
        
        b_prm = torch.tensor(0,device=device).float()
        b_corr = 0
        b_total = 0
        
        d_prm = torch.tensor(0,device=device).float()
        d_corr = 0
        d_total = 0
        
        for key, (start, end) in tl_map.items():
            
            if key == 'func':
                continue
            
            cmd = int(key.split('_')[0])
            typ = key.split('_')[1]

            if typ == 'f':
                continue
            
            cmd_mask = (commands == cmd).float().flatten()
            
            if cmd_mask.sum() == 0:
                continue
            
            if typ == 'i' or typ == 'sq' or typ == 'sym':
                with torch.no_grad():
                    ktar = torch.argmax(tar[:,:,start:end], dim=2).flatten()
                    kpout = torch.argmax(out[:,:,start:end], dim=2).flatten()
                                        
                    d_corr += (
                        (kpout == ktar).float() * cmd_mask                        
                    ).sum().item() * 1.0
                    d_total += cmd_mask.sum().item()
                    
                d_prm += (celoss(                    
                    out[:,:,start:end].view(-1, end-start),
                    ktar
                ) * cmd_mask).sum()
                                
            elif typ == 'b':
                with torch.no_grad():
                    ktar = tar[:, :, start:end].reshape(-1, end-start)
                    kpout = (out[:,:, start:end].reshape(-1, end-start) >= 0).float()
                    b_corr += (
                        (kpout == ktar).float() * cmd_mask.unsqueeze(-1)
                    ).sum().item() * 1.0
                    b_total += cmd_mask.sum().item() * (end-start)

                b_prm += (closs(
                    out[:,:,start:end].reshape(-1, end-start),
                    ktar
                ) * cmd_mask.unsqueeze(-1)).sum()
                
        result_map['b_prm'] = b_prm 
        result_map['b_corr'] = b_corr + 1e-8 
        result_map['b_total'] = b_total + 1e-8
        result_map['d_prm'] = d_prm
        result_map['d_corr'] = d_corr + 1e-8
        result_map['d_total'] = d_total + 1e-8

        result_map['f_norm'] = (loss_config['f_prm'] * (fprm_weight.sum() + 1e-8)).item() *1.0
        
        return result_map

    def getStartLine(self):
        l = torch.zeros(1,1,self.metadata['tl_size'],device=device).float()
        l[0,0,0] = 1.0
        return l
    
    def decode_line(self, line):
        _cparam = torch.zeros(self.metadata['max_cparams'], device=device).float()
        _dparam = torch.zeros(self.metadata['max_dparams'], device=device).long()   
        fstart, fend = self.metadata['tl_map']['func']
        cmd = line[fstart:fend].argmax().item()

        float_preds = []
        bool_preds = []

        tl_map = self.metadata['tl_map']
        
        if f'{cmd}_f' in tl_map:
            fstart, fend = tl_map[f'{cmd}_f']
            float_preds = line[fstart:fend].tolist()

        if f'{cmd}_b' in tl_map:
            bstart, bend = tl_map[f'{cmd}_b']
            bool_preds = line[bstart:bend].tolist()
                    
        for i,prm in enumerate(self.metadata['cparam_map'][cmd]):
            if prm == 'f':
                v = float_preds.pop(0)
            elif prm == 'b':
                v = bool_preds.pop(0)
            
            _cparam[i] = v

        for i, prm in enumerate(self.metadata['dparam_map'][cmd]):
            istart, iend = tl_map[f'{cmd}_{prm}_{i}']
            v = torch.argmax(line[istart:iend]).item()
            _dparam[i] = v

        return cmd, _cparam, _dparam
            
    def split_lines(self, lines):
        p_func = []
        p_cparam = []
        p_dparam = []
        for line in lines:
            _f, _c, _d = self.decode_line(line)
            p_func.append(_f)
            p_cparam.append(_c)
            p_dparam.append(_d)
        return p_func, p_cparam, p_dparam

    def eval_forward(self, inp_seq, code, code_start, bb_dims, hier_ind):
        bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)
            
        hier_oh = torch.zeros(1, inp_seq.shape[1], MAX_DEPTH).to(device)
        hier_oh[0, :, min(hier_ind, 2)] = 1.0
                
        inp = self.inp_net(
            torch.cat(
                (inp_seq, bb_dims, hier_oh), dim=2)
        )
        
        gru_out, h = self.gru(inp, code.view(1,1,-1))

        out = torch.zeros(inp_seq.shape, device=device).float()

        commands = None
            
        for net in self.net_list:                
            
            if net.func is not None:
                assert commands is not None
                if net.func != commands:
                    continue
                
            if net.line_cond is not None:
                line_cond = out[:,:,net.line_cond[0]:net.line_cond[1]]
            else:
                line_cond = torch.zeros(inp_seq.shape[0], inp_seq.shape[1], 0, device=device)
                    
            if net.bb_cond is True:
                bb_cond = bb_dims
            else:
                bb_cond = torch.zeros(inp_seq.shape[0], inp_seq.shape[1], 0, device=device)
                                    
            raw_out = net(torch.cat((
                gru_out, line_cond, bb_cond
            ), dim=2))
                
            if net._type == 'func':
                cmd = torch.argmax(raw_out.squeeze()).item()
                out[0,0, net.start+cmd] = 1.0
                assert commands == None
                commands = cmd

            elif net._type == 'disc':
                m = torch.argmax(raw_out.squeeze()).item()
                out[0,0, net.start+m] = 1.0

            elif net._type == 'b':
                r = (raw_out.squeeze() >= 0.).float()
                out[0,0,net.start:net.end] = r

            elif net._type == 'f':
                bb_max = bb_cond.max().item()
                r = torch.clamp(raw_out.squeeze(), 0.0, 10.)
                out[0,0,net.start:net.end] = r
                    
        double_enc = torch.cat((
            gru_out, code_start.repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
            
        child_pred = self.child_net(
            double_enc
        )
                
        next_codes = self.next_code_net(
            double_enc                        
        ).view(inp_seq.shape[0], inp_seq.shape[1], self.max_children, -1)                
        
        return out, next_codes, child_pred, h
        
    def train_progs(self, batch, codes, loss_config):        
        result_map = {key:0. for key in loss_config}
        
        bbox_target = torch.stack([b['bbox_gt'] for b in batch], dim=0)
        bbox_pred = self.bb_net(codes)    
        bbox_loss = (bbox_target - bbox_pred).abs().sum() 
        result_map['bbox'] = bbox_loss

        qp = batch
        qe = codes
        qbb = bbox_target
        qhi = torch.zeros(len(batch), device=device).long()
        
        while len(qp) > 0:
            bs = min(len(batch), len(qp))

            bprogs = qp[:bs]
            bencs = qe[:bs]
            bbb = qbb[:bs]
            bhi = qhi[:bs]

            qp = qp[bs:]
            qe = qe[bs:]
            qbb = qbb[bs:]
            qhi = qhi[bs:]
            
            inp_seq, tar_seq, seq_weights, fprm_weights, children, \
                child_targets, child_weights, lnext_inds, cnext_inds = getBatchDecData(bprogs)
            
            pout, pnext, pchild = self.train_forward(
                inp_seq, bencs, bbb, bhi, tar_seq
            )
                                            
            _result = self.calc_loss(
                pout,
                pchild,
                tar_seq,
                child_targets,
                seq_weights,
                fprm_weights,
                child_weights
            )

            for key in _result:
                if key in result_map:
                    result_map[key] += _result[key]
                else:
                    result_map[key] = _result[key]

            _qp = []
            _qe = []
            _qbb = []
            _qhi = []

            for ((li, lj, lk), (ci, cj)) in zip(lnext_inds, cnext_inds):
                _qp.append(children[ci][cj])
                _qe.append(pnext[li, lj, lk])
                _qbb.append(children[ci][cj]['bbox_gt'])
                _qhi.append(bhi[li]+1)
                
            if len(_qp) > 0:
                qe = torch.cat((qe, torch.stack(_qe)), dim = 0)
                qbb = torch.cat((qbb, torch.stack(_qbb).to(device)), dim = 0)
                qp += _qp
                qhi = torch.cat((qhi, torch.stack(_qhi)), dim = 0)

        return result_map


def writeConfigFile(args):
    os.system(f'mkdir {outpath} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/test > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/test > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/models > /dev/null 2>&1')
    with open(f"{outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f"{args}\n")

def get_max_sq_len(nodes):
    m = 0
    q = [n for _,n in nodes]    
    while len(q) > 0:
        n = q.pop(0)
        l = n['func_gt'].shape[0]
        m = max(m, l)            
        for c in n['children']:
            if len(c) > 0:
                q.append(c)    
    return m

def addTargets(node, metadata):
    gt_funcs = node['func_gt']
    gt_cparams = node['cparam_gt']
    gt_dparams = node['dparam_gt']
        
    node['bbox_gt'] = torch.from_numpy(node['cparam_gt'][1][:3]).float().to(device)
    
    seq = torch.zeros(SEQ_LEN, metadata['tl_size'], device=device).float()
    fprm_weight = torch.zeros(SEQ_LEN, metadata['tl_size'], device=device).float()
    seq_mask = torch.zeros(SEQ_LEN, device=device).float()
    seq_end = 0
    for i, fn in enumerate(gt_funcs.tolist()):
        seq_mask[i] = 1.0
        seq_end = i
        line = torch.zeros(metadata['tl_size'], device=device).float()
        weight = torch.zeros(metadata['tl_size'], device=device).float()
        
        line[fn] = 1.0

        float_vals = []
        bool_vals = []
        
        for j, tp in enumerate(metadata['cparam_map'][fn]):
            if tp == 'f':
                float_vals.append(gt_cparams[i][j].item())
            elif tp == 'b':
                bool_vals.append(gt_cparams[i][j].item())
            else:
                assert False, f'bad type {tp}'

        if len(float_vals) > 0:
            start, end = metadata['tl_map'][f'{fn}_f']
            line[start:end] = torch.tensor(float_vals, device=device)
            weight[start:end] = 1.0
            
        if len(bool_vals) > 0:
            start, end = metadata['tl_map'][f'{fn}_b']
            line[start:end] = torch.tensor(bool_vals, device=device)
            
        for j, prm in enumerate(metadata['dparam_map'][fn]):
            tar = int(gt_dparams[i][j].item())
            start, end = metadata['tl_map'][f'{fn}_{prm}_{j}']            
            line[start+tar] = 1.0            
        
        seq[i] = line
        fprm_weight[i] = weight
        
    node['seq'] = seq
    node['fprm_weight'] = fprm_weight
    node['seq_end'] = torch.tensor([seq_end],device=device).long()
    node['seq_mask'] = seq_mask
    
    child_target = torch.zeros(SEQ_LEN, metadata['max_children'], device=device).float()
    child_weight = torch.zeros(SEQ_LEN, metadata['max_children'], device=device).float()
    child_enc_mask = torch.zeros(SEQ_LEN, device = device).float()
    
    child_to_ln_map = {}    
    
    for i, inds in enumerate(node['child_gt']):
        for j, cind in enumerate(inds):
            child_weight[i, j] = 1.0
            child_to_ln_map[cind] = (i, j)
            if len(node['children'][cind]) > 0:
                child_target[i,j] = 1.0
                child_enc_mask[i] = 1.0
                
    node['child_target'] = child_target
    node['child_weight'] = child_weight
    node['child_to_ln_map'] = child_to_ln_map
    node['child_enc_mask'] = child_enc_mask
    node['exp_inds'] = []
    for i, child in enumerate(node['children']):
        if len(child) > 0:
            node['exp_inds'].append(i)
            addTargets(child, metadata)
            
    gt_cparams = node['cparam_gt']
    gt_dparams = node['dparam_gt']
    
    bool_target = torch.zeros(gt_funcs.shape[0], metadata['max_cparams'], device=device)
    bool_mask = torch.zeros(gt_funcs.shape[0], metadata['max_cparams'], device=device)
    float_target = torch.zeros(gt_funcs.shape[0], metadata['max_cparams'], device=device)
    float_mask = torch.zeros(gt_funcs.shape[0], metadata['max_cparams'], device=device)
        
    for i, tf in enumerate(gt_funcs):
        for j, tp in enumerate(metadata['cparam_map'][tf]):
            if tp == 'f':
                float_target[i][j] = gt_cparams[i][j].item()
                float_mask[i][j] = 1.0
            elif tp == 'b':
                bool_target[i][j] = gt_cparams[i][j].item()
                bool_mask[i][j] = 1.0
            else:
                assert False, f'bad type {tp}'
    
    disc_target = torch.zeros(gt_funcs.shape[0], metadata['max_dparams'], device=device).long()
    disc_mask = torch.zeros(gt_funcs.shape[0], metadata['max_dparams'], device=device)

    for i, tf in enumerate(gt_funcs):
        for j, _ in enumerate(metadata['dparam_map'][tf]):
            disc_target[i][j] = gt_dparams[i][j].item()
            disc_mask[i][j] = 1.0

    node['e_func_gt'] = torch.tensor(node['func_gt'], device=device).long()
    node['e_cparam_gt'] = torch.tensor(node['cparam_gt'], device=device).float()
    node['e_dparam_gt'] = torch.tensor(node['dparam_gt'], device=device).long()
    node['e_bool_target'] = bool_target.float()
    node['e_float_target'] = float_target.float()
    node['e_disc_target'] = disc_target.long()    
    node['e_bool_mask'] = bool_mask.float()
    node['e_float_mask'] = float_mask.float()
    node['e_disc_mask'] = disc_mask.float()

            
def _col(samples):
    return samples

def _bcol(samples):    
    return samples

# Full encoder + decoder training logic for a single program (i.e. a batch)
def model_train(batch, encoder, decoder, enc_opt, dec_opt, loss_config):
        
    codes = get_pc_encodings([b[2] for b in batch], encoder)    
    shape_result = decoder.train_progs([b[1] for b in batch], codes, loss_config)    
    
    loss = 0.
    
    for key in loss_config:
        loss += (loss_config[key] * shape_result[key]) / len(batch)
        shape_result[key] = (loss_config[key] * shape_result[key].item()) / len(batch)
       
    if torch.is_tensor(loss) and enc_opt is not None and dec_opt is not None:
        dec_opt.zero_grad()
        enc_opt.zero_grad()                    
        loss.backward()
        dec_opt.step()
        enc_opt.step()

    shape_result['loss'] = loss.item()
    
    return shape_result

def model_train_results(dataset, encoder, decoder, enc_opt, dec_opt, loss_config):

    if enc_opt is not None and dec_opt is not None:
        decoder.train()
        encoder.train()
    else:
        decoder.eval()
        encoder.eval()    
    
    ep_result = {}
    bc = 0.
    for batch in dataset:
        bc += 1. 
        batch_result = model_train(
            batch, encoder, decoder, dec_opt, enc_opt, loss_config
        )
        for key in batch_result:                        
            if key not in ep_result:                    
                ep_result[key] = batch_result[key]
            else:
                ep_result[key] += batch_result[key]

    ep_result['batch_count'] = bc
                
    return ep_result

def model_eval(
    eval_train_dataset, eval_val_dataset, eval_test_dataset, encoder, decoder, exp_name, epoch, num_write, metadata
):
    eval_results = {}

    for name, dataset in [
        ('train', eval_train_dataset), ('val', eval_val_dataset), ('test', eval_test_dataset)
    ]:
        if len(dataset) == 0:
            eval_results[name] = {}
            continue
        
        named_results = {
            'count': 0.,
            'miss_hier_prog': 0.,
            'no_norm': 1.0
        }
        
        recon_sets = []
        
        for batch in dataset:
            assert len(batch) == 1, 'batch size 1'
            shape = batch[0]                
            named_results['count'] += 1.
                           
            code = get_pc_encodings([shape[2]], encoder)    
            
            node = sv.sem_eval_prog(decoder, code.squeeze())

            try:
                shape_result = getShapeEvalResult(node, shape[1])
                shape_result['bbox'] = (node['bb_dims'] - shape[1]['bbox_gt']).abs().sum().item()
            except Exception as e:
                if VERBOSE:
                    print(f"FAILED SHAPE EVAL RESULT WITH {e}")
                shape_result = {}
                            
            for key in shape_result:
                if key not in named_results:
                    named_results[key] = shape_result[key]
                else:
                    named_results[key] += shape_result[key]

            try:
                fillProgram(
                    metadata['dsl'],
                    node,
                    metadata,
                    FUNC_PRED_FIELD,
                    CPARAM_PRED_FIELD,
                    DPARAM_PRED_FIELD,
                )
                recon_sets.append((node, shape[1], shape[0], shape[2]))
                
            except Exception as e:
                if VERBOSE:
                    print(f"Failed Recon Program with {e}")                                    
                named_results[f'miss_hier_prog'] += 1.
                                                        
        # For reconstruction, get metric performance
        recon_results, recon_misses = infer_recon_metrics.recon_metrics(
            recon_sets, outpath, exp_name, name, epoch, VERBOSE, num_write + 1
        )

        for key in recon_results:
            named_results[key] = recon_results[key]
    
        named_results[f'miss_hier_prog'] += recon_misses
        
        named_results[f'prog_creation_perc'] = (
            named_results[f'count'] - named_results[f'miss_hier_prog']
        ) / named_results[f'count']

        eval_results[name] = named_results

                                
    return eval_results

def print_train_results(result, exp_name):
    
    res = ""
    for name, key, norm_key in TRAIN_LOG_INFO:            
        if key in result:
            res += f"    {name} : {round(result[key] / (result[norm_key]+1e-8), 2)}\n"
            
    utils.log_print(res, f"{outpath}/{exp_name}/log.txt")


def print_eval_results(result, exp_name):    
    res = ""
    for name, key, norm_key in EVAL_LOG_INFO:            
        if key in result:
            res += f"    {name} : {round(result[key] / (result[norm_key]+1e-8), 4)}\n"
            
    utils.log_print(res, f"{outpath}/{exp_name}/log.txt")
    

def make_train_plots(train_result, val_result, train_plots, aepochs, exp_name):
    for name, key, norm_key in TRAIN_LOG_INFO:                
        for rname, result in [('train', train_result), ('val', val_result)]:            
            if key not in result:
                continue
            res = result[key] / (result[norm_key]+1e-8)
            if name not in train_plots[rname]:
                train_plots[rname][name] = [res]
            else:
                train_plots[rname][name].append(res)

        if name not in train_plots['train']:
            continue
        plt.clf()                    
        plt.plot(aepochs, train_plots['train'][name], label='train')
        if name in train_plots['val']:
            plt.plot(aepochs, train_plots['val'][name], label='val')            
        plt.legend()        
        plt.grid()
        plt.savefig(f"{outpath}/{exp_name}/plots/train/{name}.png")

        
def make_eval_plots(eval_result, eval_plots, aepochs, exp_name):
    for name, key, norm_key in EVAL_LOG_INFO:            
        for rname, result in [
            ('train', eval_result['train']), ('val', eval_result['val']), ('test', eval_result['test'])
        ]:
            if key not in result:
                continue
            res = result[key] / (result[norm_key]+1e-8)
            if not name in eval_plots[rname]:
                eval_plots[rname][name] = [res]
            else:
                eval_plots[rname][name].append(res)

        if name not in eval_plots['train']:
            continue
        plt.clf()                    
        plt.plot(aepochs, eval_plots['train'][name], label='train')
        if name in eval_plots['val']:
            plt.plot(aepochs, eval_plots['val'][name], label='val')
        if name in eval_plots['test']:
            plt.plot(aepochs, eval_plots['test'][name], label='test')
        plt.legend()        
        plt.grid()
        plt.savefig(f"{outpath}/{exp_name}/plots/eval/{name}.png")        

        
# Helper function for keeping consistent train/val splits
def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

def get_tensor_layout(metadata):
    # Figure out size of tensor
    # map from (func, type) -> indices in tensor
    
    tl_map = {}    
    size = len(metadata['func_map'])
    tl_map['func'] = (0, size)
        
    start = size        
    size += (I_LENGTH * metadata['max_d_i_params']) \
            + (SQ_LENGTH * metadata['max_d_sq_params']) \
            + (SYM_LENGTH * metadata['max_d_sym_params'])
    
    for func, prms in metadata['dparam_map'].items():
        i_start = start
        sq_start = i_start + I_LENGTH * metadata['max_d_i_params']
        sym_start = sq_start + SQ_LENGTH * metadata['max_d_sq_params']
        for i, _typ in enumerate(prms):
            if _typ == 'i':
                opt_len = I_LENGTH
                _start = i_start
                i_start += opt_len
            elif _typ == 'sq':
                opt_len = SQ_LENGTH
                _start = sq_start
                sq_start += opt_len
            elif _typ == 'sym':
                opt_len = SYM_LENGTH
                _start = sym_start
                sym_start += opt_len
                
            tl_map[f'{func}_{_typ}_{i}'] = (
                _start,
                _start + opt_len,
            )        
            
    for func, prms in metadata['cparam_map'].items():
        nf = 0
        nb = 0
        for prm in prms:
            if 'f' in prm:
                nf += 1
            elif 'b' in prm:
                nb += 1

        if nf > 0:
            size += nf
            tl_map[f'{func}_f'] = (size-nf, size)

        if nb > 0:
            size += nb
            tl_map[f'{func}_b'] = (size-nb, size)
            
    return size, tl_map
    

# Main entry-point of modeling logic
def run_train(
        dataset_path,
        exp_name,
        max_shapes,
        epochs,
        hidden_dim,
        eval_per,
        loss_config, 
        rd_seed,
        print_per,
        num_write,
        dec_lr,
        enc_lr,
        save_per,
        category,
        batch_size
):

    random.seed(rd_seed)
    numpy.random.seed(rd_seed)
    torch.manual_seed(rd_seed)
        
    raw_data = pickle.load(open(f"{dataset_path}_train.data", "rb"))
    metadata = pickle.load(open(f"{dataset_path}_train.meta", "rb"))
    metadata['max_children'] = max([int(i) for i in metadata['num_cube_map'].values()])
    metadata['rev_func_map'] = {v:k for k, v in metadata['func_map'].items()}

    for key in ('i', 'sq', 'sym'):
        metadata[f'max_d_{key}_params'] = max([len([ __l for __l in _l if __l == key]) for _l in metadata['dparam_map'].values()])
    
    tl_size, tl_map = get_tensor_layout(metadata)
    
    metadata['tl_size'] = tl_size
    metadata['tl_map'] = tl_map
    
    all_inds = []
    all_data = []
    all_pc = []

    good_inds = []
    
    max_sq_len = get_max_sq_len(raw_data)
    print(f"Seq len: {max_sq_len}")
    global SEQ_LEN
    SEQ_LEN = max_sq_len
    
    for d in tqdm(raw_data):
        
        if len(all_inds) >= max_shapes:
            break
        if len(good_inds) > 0 and d[0] not in good_inds:
            continue

        addTargets(d[1], metadata)
        
        fillProgram(
            metadata['dsl'],
            d[1],
            metadata,
            'func_gt',
            'cparam_gt',
            'dparam_gt'
        )

        pc = numpy.load(f'pc_data/{category}/{d[0]}.pts.npy')
        tpc = torch.from_numpy(pc)
        
        all_data.append(d[1])
        all_inds.append(d[0])
        all_pc.append(tpc)
        
    samples = list(zip(all_inds, all_data, all_pc))
    
    train_ind_file = f'pc_data_splits/{category}/train.txt'
    val_ind_file = f'pc_data_splits/{category}/val.txt'
    test_ind_file = f'pc_data_splits/{category}/test.txt'
        
    train_samples = []
    val_samples = []
    test_samples = []

    train_inds = getInds(train_ind_file)
    val_inds = getInds(val_ind_file)
    test_inds = getInds(test_ind_file)

    misses = 0.

    num_parts = []
    
    for (ind, prog, pc) in samples:
        if ind in train_inds or ind in good_inds:
            train_samples.append((ind, prog, pc))
            
        elif ind in val_inds:
            val_samples.append((ind, prog, pc))

        elif ind in test_inds:
            test_samples.append((ind, prog, pc))
            
        else:
            misses += 1
            
    if len(good_inds) > 0:
        val_samples = train_samples[:1]
        test_samples = train_samples[:1]
            
    print(f"Samples missed: {misses}")
    train_num = len(train_samples)
    val_num = len(val_samples)
    test_num = len(test_samples)
    
    train_dataset = DataLoader(
        train_samples, batch_size, shuffle=True, collate_fn = _bcol
    )    
    val_dataset = DataLoader(
        val_samples, batch_size, shuffle = False, collate_fn = _bcol
    )

    num_eval = max(val_num, test_num, len(good_inds))
    
    eval_train_dataset = DataLoader(
        train_samples[:num_eval], 1, shuffle=False, collate_fn = _col
    )
    eval_val_dataset = DataLoader(
        val_samples[:num_eval], 1, shuffle = False, collate_fn = _col
    )
    eval_test_dataset = DataLoader(
        test_samples[:num_eval], 1, shuffle = False, collate_fn = _col
    )

    utils.log_print(f"Training size: {len(train_samples)}", f"{outpath}/{exp_name}/log.txt")
    utils.log_print(f"Validation size: {len(val_samples)}", f"{outpath}/{exp_name}/log.txt")
    utils.log_print(f"Test size: {len(test_samples)}", f"{outpath}/{exp_name}/log.txt")

    val_epochs = []
    train_epochs = []    
    
    train_plots = {'train': {}, 'val': {}}
    eval_plots = {'train': {}, 'val': {}, 'test': {}}
    
    encoder = PCEncoder()
    decoder = dslDecoder(
        hidden_dim,
        metadata,
    )
    encoder.to(device)
    decoder.to(device)
            
    dec_opt = torch.optim.Adam(
        decoder.parameters(),
        lr = dec_lr,
        eps = 1e-6
    )

    enc_opt = torch.optim.Adam(
        encoder.parameters(),
        lr = enc_lr,
        eps = 1e-6
    )
          
    print('training ...')
    
    for e in range(0, epochs):

        json.dump({
            'train': train_plots,            
            'eval': eval_plots,            
            'train_epochs': train_epochs,
            'val_epochs': val_epochs,            
        }, open(f"{outpath}/{exp_name}/res.json" ,'w'))

        decoder.epoch = e

        do_print = (e+1) % print_per == 0
        t = time.time()
        if do_print:
            utils.log_print(f"\nEpoch {e}:", f"{outpath}/{exp_name}/log.txt")
            
        train_result = model_train_results(
            train_dataset,
            encoder,
            decoder,
            enc_opt,
            dec_opt,
            loss_config,
        )
        
        if do_print:
            with torch.no_grad():
                val_result = model_train_results(
                    val_dataset, encoder, decoder, None, None,
                    loss_config
                )                                
                
                train_epochs.append(e)
                
                make_train_plots(train_result, val_result, train_plots, train_epochs, exp_name)
            
            utils.log_print(
                f"Train results: ", f"{outpath}/{exp_name}/log.txt"
            )
            print_train_results(train_result, exp_name)

            utils.log_print(
                f"Val results: ", f"{outpath}/{exp_name}/log.txt"
            )
            
            print_train_results(val_result, exp_name)

            utils.log_print(
                f"    Time = {time.time() - t}", f"{outpath}/{exp_name}/log.txt"
            )

        with torch.no_grad():
            if (e+1) % eval_per == 0:
            
                decoder.eval()
                encoder.eval()

                t = time.time()                

                eval_results = model_eval(
                    eval_train_dataset,                    
                    eval_val_dataset,
                    eval_test_dataset,
                    encoder,
                    decoder,
                    exp_name,
                    e,
                    num_write,
                    metadata
                )
                utils.log_print(f"Evaluation training set results:", f"{outpath}/{exp_name}/log.txt")
                print_eval_results(eval_results['train'], exp_name)                
                utils.log_print(f"Evaluation validation set results:", f"{outpath}/{exp_name}/log.txt")
                print_eval_results(eval_results['val'], exp_name)
                utils.log_print(f"Evaluation test set results:", f"{outpath}/{exp_name}/log.txt")
                print_eval_results(eval_results['test'], exp_name)
                                
                utils.log_print(f"Eval Time = {time.time() - t}", f"{outpath}/{exp_name}/log.txt")

                val_epochs.append(e)
                
                make_eval_plots(eval_results, eval_plots, val_epochs, exp_name)                
                    
            if (e+1) % save_per == 0:
                utils.log_print("Saving Models", f"{outpath}/{exp_name}/log.txt")
                torch.save(decoder.state_dict(), f"{outpath}/{exp_name}/models/decoder_{e}.pt")
                torch.save(encoder.state_dict(), f"{outpath}/{exp_name}/models/encoder_{e}.pt")


def run_eval(args):
        
    raw_data = pickle.load(open(f"{args.dataset_path}_train.data", "rb"))
    metadata = pickle.load(open(f"{args.dataset_path}_train.meta", "rb"))
    metadata['max_children'] = max([int(i) for i in metadata['num_cube_map'].values()])
    metadata['rev_func_map'] = {v:k for k, v in metadata['func_map'].items()}

    for key in ('i', 'sq', 'sym'):
        metadata[f'max_d_{key}_params'] = max([len([ __l for __l in _l if __l == key]) for _l in metadata['dparam_map'].values()])
    
    tl_size, tl_map = get_tensor_layout(metadata)
    
    metadata['tl_size'] = tl_size
    metadata['tl_map'] = tl_map

    inds = []
    pc_data = []
    gt_progs = []

    test_ind_file = f'pc_data_splits/{args.category}/test.txt'
    test_inds = getInds(test_ind_file)

    for d in tqdm(raw_data):

        if len(inds) > args.num_gen:
            break
        
        if d[0] not in test_inds:
            continue
                        
        fillProgram(
            metadata['dsl'],
            d[1],
            metadata,
            'func_gt',
            'cparam_gt',
            'dparam_gt'
        )
        inds.append(d[0])
        gt_progs.append(d[1])
        pc = numpy.load(f'pc_data/{args.category}/{d[0]}.pts.npy')
        tpc = torch.from_numpy(pc).to(device)
        pc_data.append(tpc)

    encoder = PCEncoder()
    decoder = dslDecoder(
        args.hidden_dim,
        metadata,
    )
    encoder.load_state_dict(torch.load(
        f'{args.exp_name}/models/encoder_{args.load_epoch}.pt'
    ))
    decoder.load_state_dict(torch.load(
        f'{args.exp_name}/models/decoder_{args.load_epoch}.pt'
    ))

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    outname = f'{args.exp_name}/infer_output'
    
    os.system(f'mkdir {outname}')
    os.system(f'mkdir {args.exp_name}/infer_output/objs')
    os.system(f'mkdir {args.exp_name}/infer_output/progs')
    
    for ind, pc, gtprog in tqdm(list(zip(inds, pc_data, gt_progs))):
        code = get_pc_encodings([pc], encoder)    
        node = sv.sem_eval_prog(decoder, code.squeeze())
        fillProgram(
            metadata['dsl'],
            node,
            metadata,
            FUNC_PRED_FIELD,
            CPARAM_PRED_FIELD,
            DPARAM_PRED_FIELD,
        )
        try:
            pverts, pfaces = hier_execute(node)
            gtverts, gtfaces = hier_execute(gtprog)
        except Exception:
            continue

        utils.writeObj(
            pverts, pfaces, f'{outname}/objs/{ind}_pred_prog.obj'
        )
        
        utils.writeObj(
            gtverts, gtfaces, f'{outname}/objs/{ind}_gt_prog.obj'
        )

        utils.writeHierProg(node, 'dsl_prog', f"{outname}/progs/{ind}_pred_prog.txt")
        utils.writeHierProg(gtprog, 'dsl_prog', f"{outname}/progs/{ind}_gt_prog.txt")
        
        utils.writeSPC(
            pc.cpu().numpy(), f'{outname}/objs/{ind}_gt_pc.obj'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Visual Program Induction model")
    parser.add_argument('-ds', '--dataset_path', help='Path to program data, e.g. data/shapemod_chair', type = str)
    parser.add_argument('-en', '--exp_name', help='name of experiment', type = str)
    parser.add_argument('-c', '--category', type = str, help = 'category of PartNet')     
    parser.add_argument('-ms', '--max_shapes', default = 100000, type = int, help = 'max number of shapes to train/evaluate on ')
    parser.add_argument('-e', '--epochs', default = 2000, type = int, help = 'number of epochs to run for')    
    parser.add_argument('-hd', '--hidden_dim', default = 256, type = int, help = 'hidden dimension size')
    parser.add_argument('-prp', '--print_per', default = 10, type = int, help = 'how often to print out training set statistics')
    parser.add_argument('-evp', '--eval_per', default = 50, type = int, help = 'how often to run evaluation statistics')
    parser.add_argument('-sp', '--save_per', default = 10, type = int, help = 'how often to save the model')    
    parser.add_argument('-enc_lr', '--enc_lr', default = 0.0002, type = float, help = 'encoder learning rate')
    parser.add_argument('-dec_lr', '--dec_lr', default = 0.0002, type = float, help = 'decoder learning rate')    
    parser.add_argument('-rd', '--rd_seed', default = 42, type = int, help = 'random seed')    
    parser.add_argument('-ng', '--num_gen', default = 1000, type = int, help = 'number of shapes to generate each generation period')
    parser.add_argument('-nw', '--num_write', default = 25, type = int, help = 'number of shapes to write to .obj each evaluation period')
    parser.add_argument('-f_lw', '--f_lw', default = 50., type = float, help = 'weight on loss of continuous parameters')
    parser.add_argument('-d_lw', '--d_lw', default = 1., type = float, help = 'weight on loss of discrete parameters')
    parser.add_argument('-b_lw', '--b_lw', default = 1., type = float, help = 'weight on loss of boolean parameters')
    parser.add_argument('-c_lw', '--c_lw', default = 1., type = float, help = 'weight on loss of child predictions')
    parser.add_argument('-fn_lw', '--fn_lw', default = 1., type = float, help = 'weight on loss of function predictions')        
    parser.add_argument('-b', '--batch_size', default = 32, type=int, help = 'batch size')
    parser.add_argument('-le', '--load_epoch', default = None, type=int, help = 'model epoch to load from pre-trained model')
    parser.add_argument('-m', '--mode', default = 'train', type=str, help = 'whether to train new model or generate samples from pre-trained model')
    
    args = parser.parse_args()
        
    loss_config = {
        'd_prm': args.d_lw,
        'f_prm': args.f_lw,
        'bbox': args.f_lw,
        'b_prm': args.b_lw,
        'child': args.c_lw,
        'func': args.fn_lw,        
    }

    if args.mode == 'train':
        writeConfigFile(args)                    
    
        run_train(
            dataset_path=args.dataset_path,
            exp_name=args.exp_name,
            max_shapes=args.max_shapes,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            eval_per=args.eval_per,
            loss_config=loss_config,        
            rd_seed=args.rd_seed,
            print_per=args.print_per,
            num_write=args.num_write,
            enc_lr=args.enc_lr,
            dec_lr=args.dec_lr,
            save_per=args.save_per,
            category=args.category,
            batch_size=args.batch_size
        )
    elif args.mode == 'infer':
        run_eval(args)
    else:
        print(f"Bad mode {args.mode}")
        
