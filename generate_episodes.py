import os
import random
import numpy as np
import itertools
from argparse import ArgumentParser
from tqdm import trange
import torch
from Model import Net
from Agent import Agent
import time

torch.set_num_threads(1)
#net1 = torch.jit.load('checkpoints/resnet5_int8_v3.pt')
#net2 = torch.jit.load('checkpoints/resnet5_int8_v3.pt')
net1 = Net(15, 64)
net1.load_state_dict(torch.load('checkpoints/resnet5_v4.pt', map_location='cpu'))
net2 = Net(15, 64)
net2.load_state_dict(torch.load('checkpoints/resnet5_v4.pt', map_location='cpu'))


def generate_episode(netA=None, netB=None, self_play=False, max_searches=1200, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if netA is not None:
        netA.eval()
    if netB is not None:
        netB.eval()
    agentA = Agent(max_searches=max_searches, net=netA, explore=self_play)
    agentB = Agent(max_searches=max_searches, net=netB, explore=self_play)
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

    #color = random.choice([1, -1])
    color = seed % 2 * 2 - 1
    now_playing = -1
    cycles = itertools.cycle((agentA, agentB) if color == 1 else (agentB, agentA))
    first_round = True
    episode = []
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
            #print(cur_agent.current_node.eval_value)
            #print(torch.tensor(cur_agent.chess_board.count).T)
            #if now_playing == color:
            #    #print((cur_agent.root.prob.transpose(0, 1).numpy() * 100).astype(int), cur_agent.root.eval_value)
            #    print(cur_agent.root.eval_value)
            if cur_agent.chess_board.is_ended():
                if self_play:
                    reward = 1 if cur_agent.chess_board.winner == color else -1
                    episode = [(feature, prob, -reward) for feature, prob in agentA.episode]
                    episode += [(feature, prob, reward) for feature, prob in agentB.episode]
                break
    is_won = cur_agent.chess_board.winner == color
    return is_won, episode


if __name__ == '__main__':
    #result = [generate_episode(net2, net1, False, 1)]
    #exit(0)
    parser = ArgumentParser()
    parser.add_argument('--shard', default=0, type=int)
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--num_episodes', default=600, type=int)
    parser.add_argument('--save_steps', default=50, type=int)
    parser.add_argument('--do_competition', action='store_true')
    args = parser.parse_args()
    assert args.num_episodes % args.save_steps == 0
    save_dir = 'competition_results' if args.do_competition else 'episodes_data'
    os.makedirs(save_dir, exist_ok=True)
    max_searches = 300 if args.do_competition else 1200
    episodes = []
    win = 0
    start_seed = args.start_seed + args.shard * args.num_episodes
    end_seed = start_seed + args.num_episodes
    for i in trange(start_seed, end_seed, position=args.shard, desc=f'From {start_seed} to {end_seed}'):
        is_won, episode = generate_episode(netA=net1, netB=net2, self_play=not args.do_competition, max_searches=max_searches, seed=i)
        win += is_won
        episodes += episode
        if i % args.save_steps == args.save_steps - 1:
            torch.save({'total': i + 1 - start_seed, 'win': win, 'episodes': episodes, 'avg_turns': len(episodes) / (i + 1 - start_seed)}, f'{save_dir}/episodes_{args.shard}.pt')