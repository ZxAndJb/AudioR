from tqdm import tqdm
import torch
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