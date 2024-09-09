import argparse
import time
import torch
import sys
import torch.nn.functional as F
import utils
import tabulate
import vgg
from qtorch.quant import *
from qtorch.optim import OptimLP
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower
import torchvision.models as models
import logging

num_types = ["weight", "activate", "grad", "error", "momentum"]

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dataset", type=str, default="CIFAR100", help="dataset name: CIFAR10 or IMAGENET12"
)
parser.add_argument(
    "--data_path",
    type=str,
    default=".",
    metavar="PATH",
    help='path to datasets location (default: "./data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
# parser.add_argument(
#     "--swa_start", type=int, default=200, metavar="N", help="SWALP start epoch"
# )
parser.add_argument(
    "--model",
    type=str,
    default='VGG16',
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 300)",
)

parser.add_argument(
    "--bit",
    type=int,
    default=8,
    metavar="B",
    help="number of bit to train",
)
parser.add_argument(
    "--times",
    type=int,
    default=1,
    metavar="T",
    help="times of avg",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.05,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=5e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=100, metavar="N", help="random seed (default: 1)"
)
# for num in num_types:
#     parser.add_argument(
#         "--wl-{}".format(num),  # word length in bits
#         type=int,
#         default=args.bit,
#         metavar="N",
#         help="word length in bits for {}; -1 if full precision."
#     )
parser.add_argument(
    "--rounding",
    type=str,
    default="stochastic",
    metavar="S",
    choices=["stochastic"],
    help="rounding method for {}, stochastic or nearest"
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda) # 设置cpu 和gpu的随机种子

loaders = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=8) # 加载数据集
num_classes = utils.num_classes_dict[args.dataset] # 分类的数量

# prepare quantization functions
# using block floating point, allocating shared exponent along the first dimension
number_dict = dict()
for num in num_types:
    num_wl = args.bit ###
    #### core code
    # number_dict[num] = FloatingPoint(exp=5, man=2)
    number_dict[num] = BlockFloatingPoint(wl=num_wl, dim=0)
    # print("{:10}: {}".format(num, number_dict[num]))
quant_dict = dict()
for num in ["weight", "momentum", "grad"]:
    quant_dict[num] = quantizer(
        forward_number=number_dict[num], forward_rounding=args.rounding
    )

# Build model
print("bit_size: {}".format(args.bit), "epoch_number: {}".format(args.epochs), "times_number: {}".format(args.times))
name = str(args.bit)+'_'+str(args.epochs)+"_"+str(args.times)
logging.basicConfig(filename='./result/' + name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
model_cfg = getattr(vgg, args.model)  
# *model_cfg.args **model_cfg.kwargs为空
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

# automatically insert quantization modules
model = sequential_lower(
    model,
    layer_types=["conv", "linear"],  ## type of quantization
    forward_number=number_dict["activate"],
    backward_number=number_dict["error"],
    forward_rounding=args.rounding,
    backward_rounding=args.rounding,
)

model.classifier[-1] = model.classifier[-1][0]  # removing the final quantization module
model.cuda()

# Build SWALP model
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_model.swa_n = 0
swa_model.cuda()

criterion = F.cross_entropy
optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)
# model.parameters() 提供了需要优化的模型参数。weight_decay=args.wd 是权重衰减（L2 正则化）的系数，减小模型的过拟合。
# insert quantizations into the optimization loops
optimizer = OptimLP(  
    optimizer,  # optimizer：传递了之前创建的 SGD 优化器，作为原始优化器。
    weight_quant=quant_dict["weight"],
    grad_quant=quant_dict["grad"],
    momentum_quant=quant_dict["momentum"],
)
num_parameters = sum(p.numel() for p in model.parameters())
print(f"parameter number: {num_parameters}")
for name, param in model.named_parameters():
    print(name, param.shape)

def schedule(epoch):

    t = (epoch) / args.epochs
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return factor

# learning rate schedule
scheduler = LambdaLR(optimizer, lr_lambda=[schedule])


# Prepare logging
columns = [
    "ep",
    "lr",
    "tr_loss",
    "tr_acc",
    "tr_time",
    "te_loss",
    "te_acc",
]
value_of_time = args.times
for epoch in range(args.epochs):
    train_res = utils.run_epoch_avg(
        value_of_time, loaders["train"], model, criterion, optimizer=optimizer)
    scheduler.step()
    test_res = utils.run_epoch(loaders["test"], model, criterion, phase="eval")

    values = [
        epoch + 1,
        optimizer.param_groups[0]["lr"],
        *train_res.values(),
        *test_res.values(),
    ]
    table = utils.print_table(columns, values, epoch)
    logging.info(table)
    

