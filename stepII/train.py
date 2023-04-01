import torch
import argparse
from os import path
from model import Model
from utils import load_data_train, rmse, Validation


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Model(
        x_vec_size=args.x_vector_size,
        y_vec_size=args.y_vector_size,
        layers=[25, 25]
        )
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    w_loss = args.weight_fact_syn_data
    loss_syn = torch.nn.MSELoss(reduction='sum')
    loss_data = torch.nn.MSELoss(reduction='sum')

    train_data = load_data_train(
                dataset_path="./",
                file_name="input_data.csv",
                batch_size=args.batch_size,
                Shuffle_flag=args.shuffle_data,
        )

    validation_data = load_data_train(
                dataset_path="./",
                file_name="validation_data.csv",
                batch_size=10000, # big batch size to get all validation in one batch for now
                Shuffle_flag=args.shuffle_data,
        )

    loss_vals_epoch, err_vals_epoch = [], []
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, err_vals_data, err_vals_constraint, err_vals = [], [], [], []
        for X, Y in train_data:
            X, Y = X.to(device), Y.to(device)
            X_data, X_syn, Y_data = X[:, :4], X[:, 4:], Y

            y_pred_syn = model(X_syn)
            y1_plus_y2_pred = torch.sum(y_pred_syn, dim=1)
            sum_all_X_input = torch.sum(X_syn, dim=1)
            loss_val_constraint = loss_syn(y1_plus_y2_pred, sum_all_X_input)

            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            loss_val = w_loss*loss_val_constraint + loss_val_data
            err_val_data = rmse(y_pred_data, Y_data)
            err_val_constraint = rmse(y1_plus_y2_pred, sum_all_X_input)

            # Recording
            loss_vals.append(loss_val.detach().numpy().item())
            err_vals_data.append(err_val_data)
            err_vals_constraint.append(err_val_constraint)
            err_vals.append(err_val_data + err_val_constraint)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        loss_vals_epoch.append(avg_loss)
        avg_err_data = sum(err_vals_data) / len(err_vals_data)
        avg_err_constraint = sum(err_vals_constraint) / len(err_vals_constraint)
        avg_err = sum(err_vals) / len(err_vals)
        err_vals_epoch.append(avg_err)
        if epoch % 100 == 0:
            print(f"Epoch = {epoch:04d}, Average data rmse = {avg_err_data:.3f}, Average constraint rmse "
                  f"= {avg_err_constraint:.3f}, average mean rmse = {avg_err:.3f}")

        # For now, do the validation only for the last epoch
        if epoch == args.num_epoch - 1:
            model.eval()
            verr_vals, verr_vals_constraint = [], []
            for X, Y_label in validation_data:
                X, Y_label = X.to(device), Y_label.to(device)
                X_data, Y_data = X[:, :4], Y_label
                y_pred_valid_data = model(X_data)

                # Record
                verr_vals.append(rmse(y_pred_valid_data, Y_label))

                y1_plus_y2_pred = torch.sum(y_pred_valid_data, dim=1)
                sum_all_X_input = torch.sum(X_data, dim=1)
                verr_vals_constraint.append(rmse(y1_plus_y2_pred, sum_all_X_input))

            print("------------------Y-PLOTS-----------------------")
            Validation.plot_unity(y_pred=y_pred_valid_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            Validation.plot_unity(y_pred=y_pred_valid_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            print("------------------CONSTRAINT-PLOTS-----------------------")
            Validation.plot_unity(y_pred=y1_plus_y2_pred.detach().numpy(), y_actual=torch.sum(Y_label, dim=1))
            print("------------------DONE-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epoch", type=int, default=500)
    parser.add_argument("-bs", "--batch_size", type=int, default=50)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=4)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-sh_d", "--shuffle_data", type=bool, default=True)
    parser.add_argument("-w_syn_loss", "--weight_fact_syn_data", type=float, default=10.0)

    args = parser.parse_args()

    train(args)