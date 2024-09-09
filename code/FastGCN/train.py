import argparse
import time
from qtorch.quant import fixed_point_quantize, block_quantize
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from sampler import Sampler_FastGCN, Sampler_ASGCN
from utils import load_data, get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from qtorch.optim import OptimLP
from qtorch import BlockFloatingPoint
from qtorch.quant import Quantizer, quantizer

def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    parser.add_argument('--bit', type=int, default=8,
                        help='number of bits')
    parser.add_argument('--times', type=int, default=5,
                        help='numbers of average')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            optimizer.zero_grad()
            for i in range(args.times):
                output = model(sampled_feats, sampled_adjs)
                loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss   
                loss_train.backward()
            for param in model.parameters():
                param.grad.data /= args.times
            optimizer.step()
            acc_train = accuracy(output, batch_labels)
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


import math
import pdb
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, round_bit, quant, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.round_bit = round_bit
        self.quant = quant()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        support = torch.mm(input, self.weight)
        support = self.quant(support)
        output = torch.spmm(adj, support)
        output = self.quant(output)
        if self.bias is not None:
            return output + self.quant(self.bias)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, round_bit, quant, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()
        self.round_bit = round_bit
        self.quant = quant()
        self.gc1 = GraphConvolution(round_bit, quant, nfeat, nhid)
        self.gc2 = GraphConvolution(round_bit, quant, nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        output = self.gc1(x, adj[0])
        output = self.quant(output)
        outputs1 = F.relu(output)
        outputs1 = self.quant(outputs1)
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs1 = self.quant(outputs1)
        outputs2 = self.gc2(outputs1, adj[1])
        outputs2 = self.quant(outputs2)
        return self.quant(F.log_softmax(outputs2, dim=1))
        # return self.out_softmax(outputs2)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)

if __name__ == '__main__':
    # load data, set superpara and constant

    args = get_args()
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data(args.dataset)
    name = str(args.bit)+'_'+str(args.times)
    logging.basicConfig(filename='./result/' +name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ### set bolckfloating number
    round_bit = args.bit
    our_bit = BlockFloatingPoint(wl=round_bit, dim=0)
    # define quantization functions
    weight_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    grad_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    momentum_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")
    acc_quant = quantizer(forward_number=our_bit,
                            forward_rounding="stochastic")

    # define a lambda function so that the Quantizer module can be duplicated easily
    act_error_quant = lambda : Quantizer(forward_number=our_bit, backward_number=our_bit,
                            forward_rounding="stochastic", backward_rounding="stochastic")

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")

    # data for train and test
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    # init model, optimizer and loss function

    model = GCN(round_bit, act_error_quant, nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                sampler=sampler).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"parameter number: {num_parameters}")
    loss_fn = F.nll_loss
    for name, param in model.named_parameters():
        print(name, param.shape)
    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.4f}, "
              f"train_acc: {train_acc:.4f}, "
              f"train_times: {train_time:.4f}s "
              f"test_loss: {test_loss:.4f}, "
              f"test_acc: {test_acc:.4f}, "
              f"test_times: {test_time:.4f}s")
        logging.info(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} " +
              f"train_loss: {train_loss:.4f}, " +
              f"train_acc: {train_acc:.4f}, "+
              f"test_loss: {test_loss:.4f}, "+
              f"test_acc: {test_acc:.4f}, ")
    
