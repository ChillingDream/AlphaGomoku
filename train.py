import os
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import quantization
from Model import Net
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna

num_epochs = 30
load_path = None
quant = True
name = 'resnet5'
if quant:
    name += '_int8'
num_blocks = 2


class EpisodeDataset(Dataset):
    def __init__(self, path='episodes_data/episodes.pt'):
        super().__init__()
        self.data = [(x.float(), y.view(-1), torch.tensor(z, dtype=torch.float)) for x, y, z in torch.load(path)]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

data = EpisodeDataset()

def evaluate(net, test_dataloder, gpu=True):
    losses = []
    for batch in test_dataloder:
        with torch.no_grad():
            if gpu:
                feature, prob, reward = [x.cuda(non_blocking=True) for x in batch]
            else:
                feature, prob, reward = batch
            pred_prob, pred_reward = net(feature)
            #loss = F.cross_entropy(pred_prob.flatten(1), prob) + F.mse_loss(pred_reward, reward)
            loss = -(torch.log_softmax(pred_prob.flatten(1), dim=1) * prob).sum(dim=1).mean() + F.mse_loss(pred_reward, reward)
            losses.append(loss.item())
    test_loss = np.mean(losses)
    return test_loss

def train(lr, batch_size, weight_decay, save_model, display=True):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    net = Net(15, 64, num_blocks=num_blocks).cuda()
    if load_path is not None:
        net.load_state_dict(torch.load(load_path))
    net.train()
    if quant:
        net.fuse()
        net.quantize()
        quantization.prepare_qat(net, inplace=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_data, test_data = train_test_split(data, test_size=5000, random_state=42)
    train_dataloder = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloder = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
    min_test_loss = 1e10
    for epoch in range(num_epochs):
        losses = []
        net.cuda()
        net.train()
        for batch in tqdm(train_dataloder, disable=not display):
            feature, prob, reward = [x.cuda(non_blocking=True) for x in batch]
            pred_prob, pred_reward = net(feature)
            #if epoch == 1:
            #    print(torch.softmax(pred_prob[0], dim=0).detach().cpu().numpy().round(3))
            #    print(prob[0].view(15, 15).detach().cpu().numpy().round(3))
            #loss = F.cross_entropy(pred_prob.flatten(1), prob) + F.mse_loss(pred_reward, reward)
            loss = -(torch.log_softmax(pred_prob.flatten(1), dim=1) * prob).sum(dim=1).mean() + F.mse_loss(pred_reward, reward)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        #if epoch == 1:
        #    exit(0)
        train_loss = np.mean(losses)
        if quant:
            quantized_net = quantization.convert(net.cpu().eval(), inplace=False)
            test_loss = evaluate(quantized_net.eval(), test_dataloder, gpu=False)
        else:
            test_loss = evaluate(net.eval(), test_dataloder)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            if save_model:
                if quant:
                    quantized_net = torch.jit.trace(quantized_net, torch.randn(1, Net.num_channels, 15, 15))
                    torch.jit.save(quantized_net, f'checkpoints/new_{name}.pt')
                else:
                    torch.save(net.state_dict(), f'checkpoints/new_{name}.pt')
        if display:
            print(f'Epoch: {epoch}. train_loss: {train_loss}, test_loss: {test_loss}')
    return min_test_loss


def objective(trial):
    batch_size = trial.suggest_int('batch_size', 64, 512)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, step=1e-4)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, step=1e-5)
    return train(lr=lr, batch_size=batch_size, weight_decay=weight_decay, save_model=False, display=False)


os.makedirs('checkpoints', exist_ok=True)
#train(6e-3, 156, 3e-3, True)
#exit(0)
study = optuna.create_study(study_name='train', direction='minimize')
study.optimize(objective, n_trials=5)
os.makedirs(f'results', exist_ok=True)
with open(f'results/{name}_result.txt', 'w') as f:
    print(study.best_params)
    print(study.best_params, file=f)
    print(study.best_trial, file=f)
    print(study.best_trial.value)
    print(study.best_trial.value, file=f)
train(**study.best_params, save_model=True)
fig = optuna.visualization.plot_param_importances(study)
fig.write_image(f'results/{name}_param_importances.png', format='png')
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image(f'results/{name}_optimization_history.png', format='png')
optuna.visualization.plot_slice(study)
fig.write_image(f'results/{name}_slice.png', format='png')
