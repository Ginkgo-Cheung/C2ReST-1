import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from maml import STMAML
from tqdm import tqdm


def train_epoch(train_dataloader):
    train_losses = []
    for step, (data, A_wave) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        A_wave = A_wave.to(device=args.device)
        A_wave = A_wave.float()
        data = data.to(device=args.device)
        loss_predict = loss_criterion(out, data.y)
        loss_reconsturct = loss_criterion(graph, A_wave)
        loss = loss_predict + loss_reconsturct
        loss.backward()
        optimizer.step()
        # print("loss_predict: {}, loss_reconsturct: {}".format(loss_predict.detach().cpu().numpy(), loss_reconsturct.detach().cpu().numpy()))
        train_losses.append(loss.detach().cpu().numpy())
    return sum(train_losses) / len(train_losses)


def test_epoch(test_dataloader):
    with torch.no_grad():
        model.eval()
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=args.device)
            data = data.to(device=args.device)
            out, _ = model(data, A_wave)
            if step == 0:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label


print(time.strftime('%Y-%m-%d %H:%M:%S'), "target_days = ", args.target_days)

if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    target_dataset = traffic_dataset(data_args, task_args, "target", test_data=args.test_dataset,
                                     target_days=args.target_days)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['batch_size'], shuffle=True, num_workers=8,
                                   pin_memory=True)
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=8,
                                 pin_memory=True)

    model.finetuning(target_dataloader, test_dataloader, args.target_epochs)
    print(args.memo)
