
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def fit_model(
    model,
    dataset,
    training_params,
    device,
    criterion=nn.CrossEntropyLoss(),

):
    print(f"Training in {device}")
    print(f"Making {torch.cuda.get_device_name(0)} go brrruuummmmm....")
    dataloader = dict(
        train=DataLoader(
            dataset["train"],
            batch_size=training_params["batch_size_train"],
            pin_memory=True,
            num_workers=10,
            drop_last=True,
            shuffle=True,
        ),
        val=DataLoader(
            dataset["val"],
            batch_size=training_params["batch_size_val"],
            pin_memory=True,
            num_workers=10,
            shuffle=False,
        ),
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])
    total_loss = dict(train=[], val=[])
    for epoch in range(training_params["epochs"]):
        epoch_loss = dict(train=[], val=[])
        model.train()
        tbar = tqdm(dataloader["train"])
        for batch in tbar:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss["train"].append(loss.item())
            tbar.set_description(f"Training Loss: {np.mean(epoch_loss['train']):.4f}")

        train_loss = np.mean(epoch_loss["train"])

        model.eval()
        with torch.no_grad():
            vbar = tqdm(dataloader["val"])
            for batch in vbar:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                epoch_loss["val"].append(loss.item())
                vbar.set_description(f"Validation Loss: {np.mean(epoch_loss['val']):.4f}")

        val_loss = np.mean(epoch_loss["val"])

        total_loss["train"].append(train_loss)
        total_loss["val"].append(val_loss)
        print(
            f"Epoch: {epoch + 1}/{training_params["epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        print("==================================")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/CharRNN_{epoch+1}.pth")
    return model, total_loss