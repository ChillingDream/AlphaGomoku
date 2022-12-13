import glob
import torch
results = []
for filename in glob.glob('episodes_data/episodes_*.pt'):
    results.append(torch.load(filename))
torch.save(sum(results, []), 'episodes_data/episodes.pt')