import os, torch, logging, argparse
import models
from utils import train, test, val
from data import load_data
import datetime
from pathlib import Path

# out dir 
OUT_PATH = "results"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Name to add")
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='GCN', help='{SGC, DeepGCN, DeepGAT}')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--nhead', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# for deep model
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--residual', type=int, default=0, help='Residual connection')
# for PairNorm
# - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
parser.add_argument('--norm_mode', type=str, default='None', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
# for data
parser.add_argument('--no_fea_norm', action='store_false', default=True, help='not normalize feature' )
parser.add_argument('--missing_rate', type=int, default=0, help='missing rate, from 0 to 100' )

# Embedding relations
parser.add_argument('--difference', action='store_true', default=False, help='h1 - h2')
parser.add_argument('--abs_difference', action='store_true', default=False, help='|h1 - h2|')
parser.add_argument('--elem_product', action='store_true', default=False, help='h1 * h2')

args = parser.parse_args()

relations = {"difference": args.difference,
             "abs_difference": args.abs_difference,
             "elem_product": args.elem_product}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# logger
#filename='example.log'
logging.basicConfig(format='%(message)s', level=getattr(logging, args.log.upper())) 

# load data
data = load_data(args.data, normalize_feature=args.no_fea_norm, missing_rate=args.missing_rate,
                 cuda=torch.cuda.is_available())
nfeat = data.x.size(1)
nclass = int(data.y.max()) + 1
net = getattr(models, args.model)(nfeat, args.hid, nclass, 
                                  dropout=args.dropout, 
                                  nhead=args.nhead,
                                  nlayer=args.nlayer, 
                                  norm_mode=args.norm_mode,
                                  norm_scale=args.norm_scale,
                                  residual=args.residual,
                                  relations=relations)
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()
logging.info(net)

# train
best_acc = 0 
best_loss = 1e10

log_dir = os.path.join(OUT_PATH, args.name)
Path(log_dir).mkdir(parents=True, exist_ok=True)

file_name_prefix = args.data + "_" + args.model + "_" + str(args.nlayer) + datetime.datetime.now().strftime(
    "_%Y.%m.%d_%H.%M.%S_")

for epoch in range(args.epochs):
    train_loss, train_acc = train(net, optimizer, criterion, data)
    val_loss, val_acc = val(net, criterion, data)
    logging.debug('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f.'%
                (epoch, train_loss, train_acc, val_loss, val_acc))
    # save model

    if best_acc < val_acc:
        best_acc = val_acc
        file_name = os.path.join(log_dir, file_name_prefix + 'checkpoint-best-acc.pkl')
        torch.save(net.state_dict(), file_name)
    if best_loss > val_loss:
        best_loss = val_loss
        file_name = os.path.join(log_dir, file_name_prefix + 'checkpoint-best-loss.pkl')
        torch.save(net.state_dict(), file_name)

# pick up the best model based on val_acc, then do test

file_name = os.path.join(log_dir, file_name_prefix + 'checkpoint-best-acc.pkl')
net.load_state_dict(torch.load(file_name))
val_loss, val_acc = val(net, criterion, data)
test_loss, test_acc = test(net, criterion, data)

logging.info("-"*50)
logging.info("Vali set results: loss %.3f, acc %.3f."%(val_loss, val_acc))
logging.info("Test set results: loss %.3f, acc %.3f."%(test_loss, test_acc))
logging.info("="*50)
