#%%
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss

from utils import plot_losses, plot_trials
from get_data import get_data

def train(model, epochs = 100, batch_size = 128, show_after = 10):
    constant_x, constant_y = get_data(batch_size = 5**2, test = True)
    train_losses = []; test_losses = []
    for e in tqdm(range(1,epochs+1)):
        x, y = get_data(batch_size = batch_size, test = False)
        predicted = model(x)
        loss = F.nll_loss(predicted, torch.argmax(y,-1))
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()
        train_losses.append(loss.item())
        
        with torch.no_grad():
            x, y = get_data(batch_size = batch_size, test = True)
            predicted = model(x)
            loss = F.nll_loss(predicted, torch.argmax(y,-1))
            test_losses.append(loss.item())
            
            if(e%show_after == 0 or e==1):
                plot_losses(e, train_losses, test_losses)
                plot_trials(e, constant_x, constant_y, model(x))
                print()
# %%
