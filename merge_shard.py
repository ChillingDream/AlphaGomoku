import glob
import torch
total = 0
win = 0
episodes = []
dir = 'episodes_data'
#dir = 'competition_results'
for filename in glob.glob(f'{dir}/episodes_*.pt'):
    result = torch.load(filename)
    total += result['total']
    win += result['win']
    episodes += result['episodes']
torch.save({'total': total, 'win': win, 'episodes': episodes, 'avg_turns': len(episodes) / total}, f'{dir}/episodes.pt')
print(total, win)