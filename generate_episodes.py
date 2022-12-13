import os
import random
import itertools
from argparse import ArgumentParser
from tqdm import trange
import torch
from Model import Net
from Agent import Agent
import time

torch.set_num_threads(1)
net1 = Net(15, 64)
#net2 = Net(15, 64, num_blocks=3)
net1.load_state_dict(torch.load('resnet5.pt', map_location='cpu'))
#net2.load_state_dict(torch.load('new_model_block3.pt'))


def generate_episode(netA=None, netB=None, self_play=False, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    if netA is not None:
        netA.eval()
    if netB is not None:
        netB.eval()
    agentA = Agent(max_searches=1000, net=netA)
    agentB = Agent(max_searches=1000, net=netB)
    if self_play:
        agentA.episode = []
        agentB.episode = []
        episode = []
    ##moves = [(7, 7), (8, 8), (8, 7), (6, 8), (6, 7), (7, 8), (5, 7), (9, 8), (4, 7)]
    #moves = [(7, 7), (1, 1), (9, 7), (1, 2), (8, 6), (2, 1), (6, 7), (8, 7), (8, 8), (2, 2), (10, 7)]
    #moves = [(7, 7), (9, 7), (6, 7), (10, 7), (7, 8), (9, 8), (6, 9), (11, 8), (9, 6), (10, 9), (10, 5), (11, 10), (11, 4)]
    #for move in moves:
    #    if move == (9, 6):
    #        print('*')
    #    agentA.update_root(move)
    #    agentA.chess_board.display_board()
    #    #print(torch.tensor(agentA.chess_board.cross, dtype=torch.long).T)
    #    print(agentA.chess_board.link3)
    #    print(agentA.chess_board.link4)
    #    print(agentA.chess_board.link5)
    #    print(agentA.chess_board.cross)
    #    print(agentA.chess_board.dir)
    #    #print((agentA.root.prob.transpose(0, 1).numpy() * 100).astype(int), agentA.root.eval_value)
    #    #print(agentA.eval_value())
    #exit(0)

    color = random.choice([1, -1])
    now_playing = -1
    cycles = itertools.cycle((agentA, agentB) if color == 1 else (agentB, agentA))
    first_round = True
    for cur_agent in cycles:
        now_playing = -now_playing
        if first_round:
            move = (7, 7)
            cur_agent.update_root(move)
            first_round = False
        else:
            #st = time.time()
            #print(move)
            move = cur_agent.search(move)
            #print(color, now_playing, move, time.time() - st)
            #cur_agent.chess_board.display_board()
            #print(torch.tensor(cur_agent.chess_board.count).T)
            #if now_playing == color:
            #    #print((cur_agent.root.prob.transpose(0, 1).numpy() * 100).astype(int), cur_agent.root.eval_value)
            #    print(cur_agent.root.eval_value)
            if cur_agent.chess_board.is_ended():
                if self_play:
                    reward = 1 if cur_agent.chess_board.winner == color else -1
                    episode = [(feature, prob, -reward) for feature, prob in agentA.episode]
                    episode += [(feature, prob, reward) for feature, prob in agentB.episode]
                    return episode
                else:
                    return cur_agent.chess_board.winner == color


if __name__ == '__main__':
    #result = [generate_episode(net1, net1, True, 3)]
    #exit(0)
    os.makedirs('episodes_data', exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--shard', default=0, type=int)
    parser.add_argument('--num_episodes', default=600, type=int)
    args = parser.parse_args()
    result = []
    for i in trange(args.shard * args.num_episodes, (args.shard + 1) * args.num_episodes):
        result += generate_episode(netA=net1, netB=net1, self_play=True, seed=i)
        if i % 50 == 49:
            torch.save(result, f'episodes_data/episodes_{args.shard}.pt')
    torch.save(result, f'episodes_data/episodes_{args.shard}.pt')