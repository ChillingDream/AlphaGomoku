import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from Model import Net
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
from matplotlib import pyplot as plt

num_epochs = 30
load_path = None

data = [(x.float(), y, torch.tensor(z, dtype=torch.float)) for x, y, z in torch.load('episodes_data/episodes.pt')]

def train(lr, batch_size, weight_decay, save_model, display=True):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    net = Net(15, 64, num_blocks=3).cuda()
    if load_path is not None:
        net.load_state_dict(torch.load(net))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_data, test_data = train_test_split(data, test_size=5000, random_state=1)
    train_dataloder = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloder = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
    min_test_loss = 1e10
    for epoch in range(num_epochs):
        losses = []
        net.train()
        for batch in tqdm(train_dataloder, disable=not display):
            feature, prob, reward = [x.cuda(non_blocking=True) for x in batch]
            pred_prob, pred_reward = net(feature)
            loss = F.cross_entropy(pred_prob, prob) + F.mse_loss(pred_reward, reward)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss = np.mean(losses)

        net.eval()
        losses = []
        for batch in test_dataloder:
            with torch.no_grad():
                feature, prob, reward = [x.cuda(non_blocking=True) for x in batch]
                pred_prob, pred_reward = net(feature)
                loss = F.cross_entropy(pred_prob, prob) + F.mse_loss(pred_reward, reward)
                losses.append(loss.item())
        test_loss = np.mean(losses)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            if save_model:
                torch.save(net.state_dict(), 'new_model_block3.pt')
        if display:
            print(f'Epoch: {epoch}. train_loss: {train_loss}, test_loss: {test_loss}')
    return min_test_loss


def objective(trial):
    batch_size = trial.suggest_int('batch_size', 64, 512)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, step=1e-4)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, step=1e-5)
    return train(lr=lr, batch_size=batch_size, weight_decay=weight_decay, save_model=False, display=False)


##train(5e-4, 128, 1e-3, False)
##exit(0)
study = optuna.create_study(study_name='train', direction='minimize')
study.optimize(objective, n_trials=2)
with open('result.txt', 'w') as f:
    print(study.best_params)
    print(study.best_params, file=f)
    print(study.best_trial, file=f)
    print(study.best_trial.value)
    print(study.best_trial.value, file=f)
    #optuna.visualization.plot_param_importances(study)
    #plt.savefig('param_importances.png', format='png')
    #optuna.visualization.plot_optimization_history(study)
    #plt.savefig('optimization_history.png', format='png')
    #optuna.visualization.plot_slice(study)
    #plt.savefig('slice.png', format='png')