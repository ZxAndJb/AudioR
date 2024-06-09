import os.path
from transformers import ClapModel
import torch
from utils.datautils import AudioTextDataset, collate_fn
from torch.utils.data import DataLoader
import pandas as pd
from utils.modelutils import train, eval, save
from torch.cuda.amp import GradScaler

if __name__ == '__main__':

    model = ClapModel.from_pretrained('laion/larger_clap_general')
    data_path = "./data"
    df_path = ["train", "validation", "in_test", "out_test"]
    train_path = os.path.join(data_path, df_path[0])
    valid_path = os.path.join(data_path, df_path[1])
    in_test_path = os.path.join(data_path, df_path[2])
    out_test_path = os.path.join(data_path, df_path[3])

    train_df = pd.read_csv(os.path.join(train_path,"meta.csv"))
    valid_df = pd.read_csv(os.path.join(valid_path,"meta.csv"))
    in_test_df = pd.read_csv(os.path.join(in_test_path,"meta.csv"))
    out_test_df = pd.read_csv(os.path.join(out_test_path,"meta.csv"))

    train_ds = AudioTextDataset(train_df, train_path)
    valid_ds = AudioTextDataset(valid_df, valid_path)
    in_test_ds = AudioTextDataset(in_test_df, in_test_path)
    out_test_ds = AudioTextDataset(out_test_df, out_test_path)

    train_dataloader = DataLoader(train_ds, batch_size=3, shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=3, shuffle=True, collate_fn=collate_fn, drop_last=True)
    int_dataloader = DataLoader(in_test_ds, batch_size=3, shuffle=True, collate_fn=collate_fn, drop_last=True)
    out_dataloader = DataLoader(out_test_ds, batch_size=3, shuffle=True, collate_fn=collate_fn, drop_last=True)


    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    scaler = GradScaler()

    max_epoch = 30

    for epoch in range(max_epoch):
        if epoch> 0:
            train(model, train_dataloader, optimizer, scaler, 2, device)
        epoch_result = {}
        epoch_result["train"] = eval(model, train_dataloader, device)
        epoch_result["vali"] = eval(model, valid_dataloader, device)
        print(f"epoch {epoch + 1}, train loss: {epoch_result['train']}, valid loss: {epoch_result['vali']}")

    save(model, optimizer, "./")






