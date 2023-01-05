import torch
from torch import nn
from ChessBoard import ChessBoard
from torch import quantization


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.cnn1 = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU(True)
        self.cnn2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.relu2 = nn.ReLU(True)
        self.skip = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        h = self.relu1(self.bn1(self.cnn1(x)))
        return self.relu2(self.skip.add(x, self.bn2(self.cnn2(h))))
    
    def fuse(self):
        quantization.fuse_modules(self, ['cnn1', 'bn1', 'relu1'], inplace=True)
        quantization.fuse_modules(self, ['cnn2', 'bn2', 'relu2'], inplace=True)


class Net(nn.Module):
    num_channels = 15

    def __init__(self, board_size, hidden_size, num_blocks=2, kernel_size=3):
        super().__init__()
        self.cnn1 = nn.Conv2d(Net.num_channels, hidden_size, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, hidden_size, kernel_size) for _ in range(num_blocks)]
        )
        self.p_head = nn.Sequential(
            nn.Conv2d(hidden_size, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(board_size ** 2 * 2, board_size ** 2)
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(hidden_size, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = x.view(-1, Net.num_channels, 15, 15)
        x = self.relu1(self.bn1(self.cnn1(x)))
        x = self.res_blocks(x)
        p = self.p_head(x).view(x.shape[0], 15, 15)
        v = self.v_head(x).squeeze(1)
        p = self.dequant(p)
        v = self.dequant(v)
        return p, v
    
    def fuse(self):
        quantization.fuse_modules(self, ['cnn1', 'bn1', 'relu1'], inplace=True)
        for block in self.res_blocks:
            block.fuse()
        quantization.fuse_modules(self.p_head, ['0', '1', '2'], inplace=True)
        quantization.fuse_modules(self.v_head, ['0', '1', '2'], inplace=True)
        quantization.fuse_modules(self.v_head, ['4', '5'], inplace=True)
    
    def quantize(self):
        self.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    
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
        ).float().unsqueeze(0)
        return feature

    @staticmethod
    def preprocess_explore(chessboard: ChessBoard) -> torch.Tensor:
        board = torch.tensor(chessboard.board)
        count = torch.tensor(chessboard.count)
        def get_mask(*points_group: set):
            mask = torch.zeros_like(board)
            for points in points_group:
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
                get_mask(chessboard.par_link3[chessboard.now_playing]),
                get_mask(chessboard.par_link4[chessboard.now_playing]),
                get_mask(chessboard.cross[chessboard.now_playing],
                         chessboard.par_cross[chessboard.now_playing]),
                (count == 3) & (board == -chessboard.now_playing),
                (count == 4) & (board == -chessboard.now_playing),
                get_mask(chessboard.link3[-chessboard.now_playing]),
                get_mask(chessboard.link4[-chessboard.now_playing]),
                get_mask(chessboard.link5[-chessboard.now_playing]),
                get_mask(chessboard.par_link3[-chessboard.now_playing]),
                get_mask(chessboard.par_link4[-chessboard.now_playing]),
                get_mask(chessboard.cross[-chessboard.now_playing],
                         chessboard.par_cross[-chessboard.now_playing]),
            ),
            dim=0
        ).float().unsqueeze(0)
        return feature
    
    @staticmethod
    def normalize_prob(p: torch.tensor, vacancies: set) -> torch.tensor:
        mask = torch.zeros_like(p, dtype=torch.bool)
        for move in vacancies:
            mask[move] = True
        p = torch.where(mask, p, torch.full_like(p, -100))
        p = torch.softmax(p.view(-1), dim=0).view(p.shape)
        return p