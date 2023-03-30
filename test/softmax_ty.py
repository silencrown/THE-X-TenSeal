import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter

class ReciprocalApproximation(nn.Module):
    def __init__(self, dim_size=16):
        super(ReciprocalApproximation, self).__init__()
        self.transform = nn.Linear(1, dim_size)
        self.dense = nn.Linear(dim_size, dim_size)
        self.predict = nn.Linear(dim_size, 1)

    def forward(self, x):
        x = F.relu(self.transform(x))
        x = F.relu(self.dense(x))
        x = self.predict(x)
        return x

class SoftmaxApproximation(nn.Module):
    def __init__(self, dim_size=16):
        super(SoftmaxApproximation, self).__init__()
        self.reciprocal = ReciprocalApproximation(dim_size=dim_size)
        # self.reciprocal.requires_grad_(False)

    def forward(self, attention_score, dim):
        # (x/2+1) -> ^3 -> ReLU -> sum -> 
        t = attention_score / 2 + 1
        exp_of_score = F.relu(t * t * t)
        x = exp_of_score.sum(dim, keepdim=True).unsqueeze(-1)
        # x -> [N, L, 1, H, 1]
        x = self.reciprocal(x).squeeze(dim=-1)
        # [N, L, 1, H]
        return exp_of_score * x


def train(args, model):
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    iter_bar = tqdm(range(args.num_iter))
    for t in iter_bar:
        input_tensor = torch.randn(128, 2, 128, 128, device=device) * args.scale
        label  = nn.Softmax(dim=-1)(input_tensor)

        pred = model(input_tensor, dim=-1)

        loss = nn.MSELoss()(pred.view(-1), label.view(-1))
        # print(loss)
        writer.add_scalar("loss", loss, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_bar.set_description("loss: %.4f" % loss.item())
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output_path", type=str, default="outputs")
    args = parser.parse_args()

    model = SoftmaxApproximation()
    # state_dict = torch.load("reciprocal_model.pt", map_location="cpu")
    # model.reciprocal.load_state_dict(state_dict)
    model = train(args, model)
    model = model.cpu()
    torch.save(model.reciprocal.state_dict(), args.output_path)

    print("save to {}".format(args.output_path))

if __name__ == "__main__":
    main()