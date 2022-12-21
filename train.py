import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import quantization
from torchvision import transforms
from Model import Net
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna

torch.set_num_threads(1)
num_epochs = 50
load_path = 'checkpoints/resnet5_v3.pt'
load_path = None
quant = False
name = 'resnet5'
if quant:
    name += '_int8'
num_blocks = 2


class DiscreteRandomRotation(nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def forward(self, x):
        dice = torch.empty(len(self.degrees)).uniform_()
        return transforms.functional.rotate(x, self.degrees[dice.argmax()], transforms.functional.InterpolationMode.NEAREST, False, None, None, None)


class EpisodeDataset(Dataset):
    def __init__(self, data=None, use_transform=True):
        super().__init__()
        self.data = [(x.float().view(15, 15, 15), y.view(1, 15, 15), torch.tensor(z, dtype=torch.float)) for x, y, z in data]
        transform = torch.nn.Sequential(
            DiscreteRandomRotation([0., 90., 180., 270.]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        )
        self.transform = torch.jit.script(transform)
        self.use_tranform = use_transform
    
    def __getitem__(self, index):
        x, y, z = self.data[index]
        if self.use_tranform:
            s = torch.cat([x, y], dim=0)
            s = self.transform(s)
            x, y = s.split([x.size(0), y.size(0)])
        return x, y.view(-1), z
    
    def __len__(self):
        return len(self.data)


def read_episodes(path):
    ckpt = torch.load(path)
    if isinstance(ckpt, list):
        return ckpt
    elif 'episodes' in ckpt:
        return ckpt['episodes']
    else:
        raise ValueError('Invalid episode file')


path = 'episodes_data/episodes.pt'
path2 = 'episodes_data/history/episodes_v2.pt'
data = read_episodes(path)
if path2 is not None:
    episodes2 = read_episodes(path2)
    np.random.shuffle(episodes2)
    data += episodes2[:len(data) // 3]
train_data, test_data = train_test_split(data, test_size=5000, random_state=42)
train_ds = EpisodeDataset(train_data, use_transform=True)
test_ds = EpisodeDataset(test_data, use_transform=False)


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
    train_dataloder = DataLoader(train_ds, batch_size, num_workers=4, shuffle=True)
    test_dataloder = DataLoader(test_ds, batch_size, shuffle=False, pin_memory=True)
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


def test():
    batch_size = 512
    net = Net(15, 64, num_blocks=num_blocks).cuda()
    net.load_state_dict(torch.load(load_path))
    _, test_data = train_test_split(data, test_size=5000, random_state=42)
    test_dataloder = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
    print(evaluate(net.eval(), test_dataloder))


def objective(trial):
    batch_size = trial.suggest_int('batch_size', 128, 512)
    lr = trial.suggest_float('lr', 5e-4, 3e-2, step=5e-4)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, step=1e-4)
    return train(lr=lr, batch_size=batch_size, weight_decay=weight_decay, save_model=False, display=False)


os.makedirs('checkpoints', exist_ok=True)
#test()
train(5e-3, 319, 1e-4, True)
exit(0)
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
fig = optuna.visualization.plot_slice(study)
fig.write_image(f'results/{name}_slice.png', format='png')
