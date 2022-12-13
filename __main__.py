import json
import torch
from Agent import Agent
from Model import Net

DEBUG = False

net = Net(15, 64)
if DEBUG:
    net.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
else:
    net.load_state_dict(torch.load('data/gomoku/best_model.pt', map_location='cpu'))

agent = Agent(
    size = 15,
    win_len = 5,
    max_searches = 300,
    net=net
)

first_round = True
x,y = 0,0
while True:
    if DEBUG:
        x, y = json.loads(input())
    else:
        fullInput = json.loads(input())
        if first_round:
            requests = fullInput["requests"]
            responses = fullInput["responses"]
            x, y = requests[0]["x"], requests[0]["y"]
            first_round = False
        else:
            x = fullInput["x"]
            y = fullInput["y"]

    if x == -1:
        agent.update_root((7, 7))
        print(json.dumps({"response": {"x": 7, "y": 7}}))
    else:
        (x, y) = agent.search((x, y))
        print(json.dumps({"response": {"x": x, "y": y}}))

    print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')