# coding=utf-8
import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # main root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from tqdm import tqdm
from copy import deepcopy

import time

from utils.general import set_seed
from utils.dataloader import CustomDataLoader
from models.tsAMD import AMD


def main(args):
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # workers
    torch.set_num_threads(4)
    # set seed
    set_seed(args.seed)

    # load datasets
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )

    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()

    # load model
    model = AMD(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        n_block=args.n_block,
        patch=args.patch,
        k=args.mix_layer_num,
        c=args.mix_layer_scale,
        alpha=args.alpha,
        target_slice=data_loader.target_slice,
        norm=args.norm,
        layernorm=args.layernorm
    ).to(device)

    print(sum(p.numel() for p in model.parameters()))

    # set criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-9)

    best_loss = torch.tensor(float('inf'))
    # create checkpoint directory
    save_directory = os.path.join(args.checkpoint_dir, args.name)

    if os.path.exists(save_directory):
        import glob
        import re

        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        save_directory = f"{path}{n}"  # update path

    os.makedirs(save_directory)

    # start training
    for epoch in range(args.train_epochs):
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0
        print(f"epoch : {epoch + 1}")
        print("Train")
        pbar = tqdm(enumerate(train_data), total=len(train_data))
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            start_time = time.time()
            outputs, moe_loss = model(batch_x)
            optimizer.zero_grad()
            loss = criterion(outputs, batch_y) + moe_loss
            loss.backward()
            optimizer.step()
            end_time = time.time()
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)
            pbar.set_description(('%-10s' * 1 + '%-10.8g ' * 1) % (f'{epoch + 1}/{args.train_epochs}', train_mloss))
            iteration_time = (end_time - start_time) * 1000
            iter_time = (iter_time * i + iteration_time) / (i + 1)
            # end batch -------------------------------------------------------------
        print(f"train loss: {train_mloss.item()}, iter_time: {iter_time}")

        model.eval()
        val_mloss = torch.zeros(1, device=device)
        val_mae = torch.zeros(1, device=device)
        val_mse = torch.zeros(1, device=device)
        print("Val")
        pbar = tqdm(enumerate(val_data), total=len(val_data))

        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, moe_loss = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                mae = torch.abs(outputs - batch_y).mean()
                val_mae = (val_mae * i + mae.detach()) / (i + 1)
                mse = ((outputs - batch_y) ** 2).mean()
                val_mse = (val_mse * i + mse.detach()) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.8g' * 1) % (f'', val_mloss))

            if val_mloss < best_loss or epoch == args.train_epochs - 1:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))

        print(f"val loss: {val_mloss.item()}, val MSE: {val_mse.item()}, val MAE: {val_mae.item()}")

        # scheduler.step()

        # end epoch -------------------------------------------------------------

    # load best model
    model.load_state_dict(best_model)

    # start testing
    model.eval()

    test_mloss = torch.zeros(1, device=device)
    test_mae = torch.zeros(1, device=device)
    test_mse = torch.zeros(1, device=device)

    print("Final Test")
    pbar = tqdm(enumerate(test_data), total=len(test_data))

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, moe_loss = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)
            mae = torch.abs(outputs - batch_y).mean()
            test_mae = (test_mae * i + mae.detach()) / (i + 1)
            mse = ((outputs - batch_y) ** 2).mean()
            test_mse = (test_mse * i + mse.detach()) / (i + 1)
            pbar.set_description(('%-10.8g' * 1) % (test_mloss))
    print(f"test loss: {test_mloss.item()}, test MSE: {test_mse.item()}, test MAE: {test_mae.item()}")


def infer_extension(dataset_name):
    if dataset_name.startswith('solar'):
        extension = 'txt'
    elif dataset_name.startswith('PEMS'):
        extension = 'npz'
    else:
        extension = 'csv'
    return extension


def parse_args():
    dataset = "ETTh1"
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    # data loader
    parser.add_argument('--data',
                        type=str,
                        default=ROOT / f'../data/{dataset}.{infer_extension(dataset)}',
                        help='dataset path')
    parser.add_argument(
        '--feature_type',
        type=str,
        default='M',
        choices=['S', 'M', 'MS'],
        help=(
            'forecasting task, options:[M, S, MS]; M:multivariate predict'
            ' multivariate, S:univariate predict univariate, MS:multivariate'
            ' predict univariate'
        ),
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=ROOT / 'checkpoints',
        help='location of model checkpoints',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=f'{dataset}',
        help='save best model to checkpoints/name',
    )
    # forecasting task
    parser.add_argument(
        # 96 192 336 512 672 720
        '--seq_len', type=int, default=720, help='input sequence length'
    )
    parser.add_argument(
        # 12 for PEMS  , {96, 192, 336, 720} for others
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )
    # model hyperparameter
    parser.add_argument(
        # 1 2 3
        '--n_block',
        type=int,
        default=1,
        help='number of block for deep architecture',
    )
    parser.add_argument(
        # 0.0  0.5  1.0
        '--alpha',
        type=float,
        default=0.0,
        help='feature feature dimension',
    )
    parser.add_argument(
        '--mix_layer_num',
        type=int,
        default=3,
        help='num of mix layer',
    )
    parser.add_argument(
        '--mix_layer_scale',
        type=int,
        default=2,
        help='scale of mix layer',
    )
    parser.add_argument(
        # 4 8 16
        '--patch',
        type=int,
        default=16,
        help='fully-connected history len',
    )
    parser.add_argument(
        '--norm',
        type=bool,
        default=True,
        help='RevIN',
    )
    parser.add_argument(
        '--layernorm',
        type=bool,
        default=True,
        help='layernorm',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate'
    )
    # optimization
    parser.add_argument(
        '--train_epochs', type=int, default=10, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size of input data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00005,
        help='optimizer learning rate',
    )
    # save results
    parser.add_argument(
        '--result_path', default='result.csv', help='path to save result'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
