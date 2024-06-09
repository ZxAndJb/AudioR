import os.path
from tqdm import tqdm
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F


def train(model, data_loader, optimizer, scaler,epoch_number, device):
    model.cuda()
    model.train()
    phar = tqdm(total=100)
    for batch_idx, data in enumerate(data_loader, 0):
        data = data.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
        # Forward + backward + optimize
            output = model(**data, return_loss=True)
            loss = output["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        phar.set_description_str("Epoch: %d" % epoch_number)
        phar.update(phar.total/len(data_loader))
    phar.close()



def eval(model,data_loader, device):
    model.cuda()
    model.eval()
    eval_loss, eval_steps = 0.0, 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            data = data.to(device)
            output = model(**data, return_loss=True)
            eval_loss += output['loss']
            eval_steps += 1
    aver_l = eval_loss / max(eval_steps, 1)
    return aver_l

def save(model, optimzier, save_path):
    path = os.path.join(save_path, "checkpoint.pth")
    torch.save(model.state_dict(), optimzier.state_dict(), path)


def load(save_path,model, optimizer: Optional =None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimzier_state = torch.load(save_path, map_location=device)
    rev_state_dict = {k.replace('module.', ''): v for k, v in model_state.items()}
    model.load_state_dict(rev_state_dict)
    if not optimzier_state:
        optimizer.load_state_dict()
    return model, optimizer
