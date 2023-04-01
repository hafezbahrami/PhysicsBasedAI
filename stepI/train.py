import torch
import argparse
from matplotlib import pyplot as plt

from model import LinearModel
from utils import load_data, accuracy


def train(args):
    model = LinearModel(x_in_vec_size=args.x_vector_size, y_vec_size=args.y_vector_size, layers=[])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # loss = torch.nn.MSELoss(reduction='sum')
    loss = torch.nn.MSELoss()


    train_data = load_data(dataset_path="./", batch_size=args.batch_size, shuffle=True)
    #valid_data = XX

    loss_vals_epoch, acc_vals_epoch = [], []
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals = [], []
        for X in train_data:
            X = X.to(device)

            y_pred = model(X)
            y_label = torch.sum(X, dim=1)[:, None]
            y_label = y_label.to(device)
            loss_val = loss(y_pred, y_label)

            loss_vals.append(loss_val.detach().numpy().item())
            acc_vals.append(accuracy(y_pred, y_label))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        loss_vals_epoch.append(avg_loss)
        avg_acc = sum(acc_vals) / len(acc_vals)
        acc_vals_epoch.append(avg_acc)
        print(avg_acc, avg_loss)
    plt.plot(acc_vals_epoch)
    zz = -1

        # model.eval()
        # for X, y_label in valid_data:
        #     X, y_label = X.to(device), y_label.to(device)
            # y_pred = model(X)
            #accuracy_val = accuracy(y_pred, y_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epoch", type=int, default=400)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=2)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    train(args)


