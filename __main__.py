import json
import torch
from Agent import Agent
from Model import Net
import time

DEBUG = False

if DEBUG:
    checkpoint = torch.load('checkpoints/resnet5_v3.pt', map_location='cpu')
else:
    checkpoint = torch.load('data/gomoku/resnet5_v3.pt', map_location='cpu')

net = Net(15, 64, num_blocks=2)
net.load_state_dict(checkpoint)
net.eval()

agent = Agent(
    size = 15,
    win_len = 5,
    max_searches = 1000,
    net=net
)

x,y = 0,0
round = 0
while True:
    round += 1
    if DEBUG:
        x, y = eval(input())
    else:
        fullInput = json.loads(input())
        if round == 1:
            requests = fullInput["requests"]
            responses = fullInput["responses"]
            x, y = requests[0]["x"], requests[0]["y"]
        else:
            x = fullInput["x"]
            y = fullInput["y"]

    if x == -1:
        agent.update_root((7, 7))
        print(json.dumps({"response": {"x": 7, "y": 7}}))
    else:
        (x, y), n = agent.search((x, y), start_time=time.clock())
        print(json.dumps({"response": {"x": x, "y": y, "search_steps":n}}))

    print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')