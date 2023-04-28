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
    # optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)
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
            X_data, X_syn, Y_data = X[:, :3], X[:, 3:], Y

            y_pred_syn = model(X_syn)

            # -------------------------------------------------------
            # Actual data loss
            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            # -------------------------------------------------------
            # Taking care of constraint losses
            # In inequalities we create residuals for outside of bounds, and then will force it to zero through losses
            y1_sgn_lesser = 0.5*(torch.sgn(y_pred_syn[:, 0]-0.9) + 1.)
            y1_pred_lesser = y1_sgn_lesser * (y_pred_syn[:, 0]-0.9)
            y1_ideal_lesser = torch.zeros(y1_sgn_lesser.size())
            loss_val_constraint1 = loss_syn(y1_pred_lesser, y1_ideal_lesser)

            y2_sgn_greater = 0.5*(torch.sgn(y_pred_syn[:, 1]-0.3) - 1.)
            y2_pred_greater = y2_sgn_greater * (y_pred_syn[:, 1]-0.3)
            y2_ideal_greater = torch.zeros(y2_sgn_greater.size())
            loss_val_constraint2 = loss_syn(y2_pred_greater, y2_ideal_greater)

            # Equality constraint
            y1_plus_y2_plus_y3_pred = torch.sum(y_pred_syn, dim=1)
            sum_c3_X_input = torch.sum(X_syn, dim=1)
            loss_val_constraint3 = loss_syn(y1_plus_y2_plus_y3_pred, sum_c3_X_input)

            loss_val_constraint = (w_loss[0]*loss_val_constraint1 + w_loss[1]*loss_val_constraint2 +  w_loss[2]*loss_val_constraint3)

            loss_val = loss_val_data + loss_val_constraint
            loss_vals.append(loss_val.detach().numpy().item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Recording
            err_val_data = rmse(y_pred_data, Y_data)
            err_val_constraint1 = rmse(y1_pred_lesser, y1_ideal_lesser)
            err_val_constraint2 = rmse(y2_pred_greater, y2_ideal_greater)
            err_val_constraint3 = rmse(y1_plus_y2_plus_y3_pred, sum_c3_X_input)
            err_val_constraint = (w_loss[0]*err_val_constraint1 + w_loss[1]*err_val_constraint2 + w_loss[2]*err_val_constraint3)/sum(w_loss)

            err_vals_data.append(err_val_data)
            err_vals_constraint.append(err_val_constraint)
            err_vals.append(err_val_data + err_val_constraint)

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
                X_data, Y_data = X[:, :3], Y_label
                y_pred_valid_data = model(X_data)

                # Record
                verr_vals.append(rmse(y_pred_valid_data, Y_label))
                
                y1_vpred_sgn_lesser = 0.5*(torch.sgn((y_pred_valid_data[:, 0])-0.9) + 1.)
                y1_vpred_lesser = y1_vpred_sgn_lesser * (y_pred_valid_data[:, 0] -0.9)
                y1_vpred_ideal_lesser = torch.zeros(y1_vpred_sgn_lesser.size())
                verr_val_constraint1 = rmse(y1_vpred_lesser, y1_vpred_ideal_lesser)
                verr_vals_constraint1.append(verr_val_constraint1)

                y2_vpred_sgn_greater = 0.5*(torch.sgn(y_pred_valid_data[:, 1]-0.3) - 1.)
                y2_vpred_greater = y2_vpred_sgn_greater * (y_pred_valid_data[:, 1]-0.3)
                y2_vpred_ideal_greater = torch.zeros(y2_vpred_sgn_greater.size())
                verr_val_constraint2 = rmse(y2_vpred_greater, y2_vpred_ideal_greater)
                verr_vals_constraint2.append(verr_val_constraint2)
                
                y1_plus_y2_plus_y3_vpred = torch.sum(y_pred_valid_data, dim=1)
                sum_c3_X_val = torch.sum(X_data, dim=1)
                verr_val_constraint3 = rmse(y1_plus_y2_plus_y3_vpred, sum_c3_X_val)
                verr_vals_constraint3.append(verr_val_constraint3)

                verr_vals_constraint.append((w_loss[0]*err_val_constraint1 + w_loss[1]*err_val_constraint2 + w_loss[2]*err_val_constraint3)/sum(w_loss))

            print("------------------Y-PLOTS-----------------------")
            # Validation.plot_unity(y_pred=y_pred_valid_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            # Validation.plot_unity(y_pred=y_pred_valid_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            Validation.plot_unity(y_pred=y_pred_valid_data[:, 2].detach().numpy(), y_actual=Y_label[:, 2].numpy())
            print("------------------CONSTRAINT-PLOTS-----------------------")
            Validation.plot_unity_w_constraint(y_con = [0.9], y_pred=y_pred_valid_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            Validation.plot_unity_w_constraint(y_con = [0.3], y_pred=y_pred_valid_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            Validation.plot_unity(y_pred=y1_plus_y2_plus_y3_vpred.detach().numpy(), y_actual=sum_c3_X_val.detach().numpy())
            # Validation.plot_regular(x_con = [0.9], x=X_data[:, 0].numpy(), y_act=Y_data[:, 0].numpy(), y_pred=y_pred_valid_data[:, 0].detach().numpy())
            # Validation.plot_regular(x_con = [0.3], x=X_data[:, 1].numpy(), y_act=Y_data[:, 1].numpy(), y_pred=y_pred_valid_data[:, 1].detach().numpy())
            print("------------------DONE-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epoch", type=int, default=5001)
    parser.add_argument("-bs", "--batch_size", type=int, default=50)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=3)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=3)
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-5)
    parser.add_argument("-sh_d", "--shuffle_data", type=bool, default=True)
    parser.add_argument("-w_syn_loss", "--weight_fact_syn_data", type=float, default=[150.0, 50.0, 10.0]) # [100.0, 150.0, 50.0]

    args = parser.parse_args()

    train(args)