import torch
from torch import nn
from torch.nn import functional as F
from ChessBoard import ChessBoard


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cnn1 = nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.cnn2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_size)
    
    def forward(self, x):
        h = F.relu(self.bn1(self.cnn1(x)))
        return F.relu(x + self.bn2(self.cnn2(h)))


class Net(nn.Module):
    num_channels = 15

    def __init__(self, board_size, hidden_size, num_blocks=2):
        super().__init__()
        self.cnn1 = nn.Conv2d(Net.num_channels, hidden_size, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, hidden_size) for _ in range(num_blocks)]
        )
        self.p_head = nn.Sequential(
            nn.Conv2d(hidden_size, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2 * 2, board_size ** 2)
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(hidden_size, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.cnn1(x)))
        x = self.res_blocks(x)
        p = self.p_head(x).view(x.shape[0], *x.shape[-2:])
        v = self.v_head(x).squeeze(1)
        return p, v
    
    @staticmethod
    def preprocess(chessboard: ChessBoard) -> torch.Tensor:
        board = torch.tensor(chessboard.board)
        count = torch.tensor(chessboard.count)
        def get_mask(points: set):
            mask = torch.zeros_like(board)
            for point in points:
                mask[point] = 1
            return mask
        feature = torch.stack(
            (
                board == 0,
                board == chessboard.now_playing,
                board == -chessboard.now_playing,
                (count == 3) & (board == chessboard.now_playing),
                (count == 4) & (board == chessboard.now_playing),
                get_mask(chessboard.link3[chessboard.now_playing]),
                get_mask(chessboard.link4[chessboard.now_playing]),
                get_mask(chessboard.link5[chessboard.now_playing]),
                get_mask(chessboard.cross[chessboard.now_playing]),
                (count == 3) & (board == -chessboard.now_playing),
                (count == 4) & (board == -chessboard.now_playing),
                get_mask(chessboard.link3[-chessboard.now_playing]),
                get_mask(chessboard.link4[-chessboard.now_playing]),
                get_mask(chessboard.link5[-chessboard.now_playing]),
                get_mask(chessboard.cross[-chessboard.now_playing]),
            ),
            dim=0
        ).float()
        return feature
    
    @staticmethod
    def normalize_prob(p: torch.tensor, vacancies: set) -> torch.tensor:
        mask = torch.zeros_like(p, dtype=torch.bool)
        for move in vacancies:
            mask[move] = True
        p = torch.where(mask, p, torch.full_like(p, -100))
        p = torch.softmax(p.view(-1), dim=0).view(p.shape)
        return p