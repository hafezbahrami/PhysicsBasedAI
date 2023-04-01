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

    w_loss = args.weight_fact_syn_data
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
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
            X_data, X_syn, Y_data = X[:, :1], X[:, 1:], Y

            y_pred_syn = model(X_syn)

            y3_sgn_lesser = 0.5*(torch.sgn(y_pred_syn-0.48) + 1.)
            y3_pred_lesser = y3_sgn_lesser * y_pred_syn
            y3_ideal_lesser = torch.zeros(y3_sgn_lesser.size())
            loss_val_constraint1 = loss_syn(y3_pred_lesser, y3_ideal_lesser)

            y3_sgn_greater = 0.5*(torch.sgn(y_pred_syn) - 1.)
            y3_pred_greater = y3_sgn_greater * y_pred_syn
            y3_ideal_greater = torch.zeros(y3_sgn_greater.size())
            loss_val_constraint2 = loss_syn(y3_pred_greater, y3_ideal_greater)

            loss_val_constraint = w_loss[0]*loss_val_constraint1 + w_loss[1]*loss_val_constraint2

            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            loss_val_constraint = (w_loss[0]*loss_val_constraint1 + w_loss[1]*loss_val_constraint2)/len(w_loss)

            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            loss_val = loss_val_data + loss_val_constraint
            loss_vals.append(loss_val.detach().numpy().item())

            # Recording
            err_val_data = rmse(y_pred_data, Y_data)
            err_val_constraint1 = rmse(y3_pred_lesser, y3_ideal_lesser)
            err_val_constraint2 = rmse(y3_pred_greater, y3_ideal_greater)
            err_val_constraint = (w_loss[0]*err_val_constraint1 + w_loss[1]*err_val_constraint2)/sum(w_loss)

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
            verr_vals, verr_vals_constraint1, verr_vals_constraint2, verr_vals_constraint3, verr_vals_constraint = [], [], [], [], []
            for X, Y_label in validation_data:
                X, Y_label = X.to(device), Y_label.to(device)
                X_data, Y_data = X[:, :1], Y_label
                y_pred_valid_data = model(X_data)

                # Record
                verr_vals.append(rmse(y_pred_valid_data, Y_label))
                
                y3_vpred_sgn_lesser = 0.5*(torch.sgn((y_pred_valid_data)-0.48) + 1.)
                y3_vpred_lesser = y3_vpred_sgn_lesser * y_pred_valid_data
                y3_vpred_ideal_lesser = torch.zeros(y3_vpred_sgn_lesser.size())
                verr_val_constraint2 = rmse(y3_vpred_lesser, y3_vpred_ideal_lesser)
                verr_vals_constraint2.append(verr_val_constraint2)

                y3_vpred_sgn_greater = 0.5*(torch.sgn(y_pred_valid_data-0.0) - 1.)
                y3_vpred_greater = y3_vpred_sgn_greater * y_pred_valid_data
                y3_vpred_ideal_greater = torch.zeros(y3_vpred_sgn_greater.size())
                verr_val_constraint1 = rmse(y3_vpred_greater, y3_vpred_ideal_greater)
                verr_vals_constraint1.append(verr_val_constraint1)

                verr_vals_constraint.append((w_loss[0]*err_val_constraint1 + w_loss[1]*err_val_constraint2)/sum(w_loss))

            print("------------------Y-PLOTS-----------------------")
            Validation.plot_regular(x_con = [0.0, 0.48],x=X_data.numpy(), y_act=Y_data.numpy(), y_pred=y_pred_valid_data.detach().numpy())
            print("------------------DONE-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epoch", type=int, default=5001)
    parser.add_argument("-bs", "--batch_size", type=int, default=50)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=1)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("-sh_d", "--shuffle_data", type=bool, default=True)
    parser.add_argument("-w_syn_loss", "--weight_fact_syn_data", type=float, default=[10.0, 10.0])

    args = parser.parse_args()

    train(args)