import os
import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mlflow
from fastai.vision.all import *

class Module(nn.Module):
    def training_step(self, batch, criterion):
        images, labels = batch
        out = self(images)
        loss = self.calculate_loss(out, labels, criterion)
        return loss, 0
    
    def validation_step(self, batch, criterion):
        images, labels = batch
        out = self(images)
        loss = self.calculate_loss(out, labels, criterion)
        return {'val_loss': loss.detach()}

    def calculate_loss(self, out, labels, criterion):
        if criterion == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()(out, labels)
        elif criterion == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()(out, labels.float())
        elif criterion == "FocalLoss":
            alpha, gamma = 0.25, 2  # Focal loss parameters, adjust as needed
            ce_loss = nn.CrossEntropyLoss()(out, labels)
            pt = torch.exp(-ce_loss)
            return (alpha * (1 - pt) ** gamma * ce_loss).mean()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        return {'val_loss': torch.stack(batch_losses).mean().item()}

    def epoch_end(self, epoch, result):
        logging.info(f"Epoch [{epoch}], Train Loss: {result['train_loss']:.10f}, Val Loss: {result['val_loss']:.10f}")


@torch.no_grad()
def evaluate(model, val_loader, criterion):
    model.eval()
    outputs = [model.validation_step(batch, criterion) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list, tuple)) else data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-7, pth_name="checkpoint.pth", lr=0.0001):
        self.patience = patience
        self.counter = 0
        self.lr = lr
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(os.getcwd(), "models", pth_name)

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None or score <= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        logging.info(f"Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}). Saving model...")
        torch.save(model.state_dict(), self.path + ".pth")
        self.val_loss_min = val_loss

def fit(epochs, lr, model_generator, train_loader, val_loader, opt_func, patience, criterion, 
        pth_name, ml_flow, log_desc, save_path, path_pth, batch_size, load_w):
    device = get_default_device()
    model = model_generator.model
    if load_w:
        model.load_state_dict(torch.load(path_pth))
    model = to_device(model, device)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    early_stopping = EarlyStopping(patience=patience, pth_name=pth_name, lr=lr)
    optimizer = get_optimizer(opt_func, model.parameters(), lr)

    history = []
    for epoch in range(epochs):
        logging.info(f"Epoch [{epoch}/{epochs-1}] started with lr = {early_stopping.lr:.10f}")
        
        train_losses = train_one_epoch(model, train_loader, optimizer, criterion, model_generator, ml_flow, epoch)
        result = evaluate(model, val_loader, criterion)
        result['train_loss'] = torch.stack(train_losses).mean().item()

        model.epoch_end(epoch, result)
        history.append(result)

        early_stopping(result['val_loss'], model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    plot_training_progress(history, save_path, pth_name)
    return history, model

def get_optimizer(opt_func, parameters, lr):
    if opt_func == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr, alpha=0.9)
    elif opt_func == 'SGD':
        return torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    elif opt_func == 'Adam':
        return torch.optim.Adam(parameters, lr)
    elif opt_func == 'AdamW':
        return torch.optim.AdamW(parameters, lr)
    elif opt_func == 'Adagrad':
        return torch.optim.Adagrad(parameters, lr)
    else:
        raise ValueError(f"Unknown optimizer function: {opt_func}")

def train_one_epoch(model, train_loader, optimizer, criterion, model_generator, ml_flow, epoch):
    model.train()
    train_losses = []
    for _indx, batch in enumerate(tqdm(train_loader, desc='Epoch')):
        optimizer.zero_grad()
        loss, _ = model.training_step(batch, criterion)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)

        if ml_flow and _indx % 10 == 0:
            mlflow.log_metric(f"batch_loss_epoch_{epoch}", loss.item())
        torch.cuda.empty_cache()
    return train_losses

def plot_training_progress(history, save_path, pth_name):
    epochs = range(1, len(history) + 1)
    val_loss = [entry['val_loss'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]

    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.plot(epochs, train_loss, 'g-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{pth_name}_loss_plot.png'))
    plt.close()

