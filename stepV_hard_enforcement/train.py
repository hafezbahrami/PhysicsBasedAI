import torch
import argparse
from os import path
from model import Model
from utils import load_data_train, rmse, projection_matrix, Validation


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    G_Indep, G_Dep, Bias = projection_matrix(
        train_data = train_data,
        input_size= args.x_vector_size,
        output_size = args.y_vector_size,
        batch_size = args.batch_size,
        Constraint_Coefs=None,
    )

    model = Model(
        x_vec_size=args.x_vector_size,
        y_vec_size=args.y_vector_size,
        layers=[25, 25],
        indep_g=G_Indep, 
        dep_g=G_Dep, 
        con_bias=Bias
        )
    model.to(device)

    w_loss = args.weight_fact_syn_data

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)
    loss_data = torch.nn.MSELoss(reduction='sum')
    loss_cons = torch.nn.MSELoss(reduction='sum')

    loss_vals_epoch, err_vals_epoch = [], []
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, err_vals = [], []
        for X, Y in train_data:
            X_data, Y_data = X.to(device), Y.to(device)
            
            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            y3_sgn_lesser = 0.5*(torch.sgn(y_pred_data[:, 2]-0.48) + 1.)
            y3_pred_lesser = y3_sgn_lesser * (y_pred_data[:, 2]-0.48)
            y3_ideal_lesser = torch.zeros(y3_sgn_lesser.size())
            loss_val_constraint1 = loss_cons(y3_pred_lesser, y3_ideal_lesser)

            y3_sgn_greater = 0.5*(torch.sgn(y_pred_data[:, 2]-0.0) - 1.)
            y3_pred_greater = y3_sgn_greater * (y_pred_data[:, 2]-0.0)
            y3_ideal_greater = torch.zeros(y3_sgn_greater.size())
            loss_val_constraint2 = loss_cons(y3_pred_greater, y3_ideal_greater)

            loss_val = loss_val_data + w_loss[0]*loss_val_constraint1 + w_loss[1]*loss_val_constraint2
            loss_vals.append(loss_val.detach().numpy().item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Recording
            err_val_data = rmse(y_pred_data, Y_data)
            err_vals.append(err_val_data)

        avg_loss = sum(loss_vals) / len(loss_vals)
        loss_vals_epoch.append(avg_loss)
        avg_err = sum(err_vals) / len(err_vals)
        err_vals_epoch.append(avg_err)
        if epoch % 100 == 0:
            print(f"Epoch = {epoch:04d}, Average mean rmse = {avg_err:.3f}")

        # For now, do the validation only for the last epoch
        if epoch == args.num_epoch - 1:
            model.eval()
            verr_vals= []
            for X, Y_label in validation_data:
                X_data, Y_data = X.to(device), Y_label.to(device)
                y_pred_valid_data = model(X_data)

                # Record
                verr_vals.append(rmse(y_pred_valid_data, Y_label))

            print("------------------Y-PLOTS-----------------------")
            # Validation.plot_unity_w_constraint(y_con = [], y_pred=y_pred_valid_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            # Validation.plot_unity_w_constraint(y_con = [], y_pred=y_pred_valid_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            # Validation.plot_unity_w_constraint(y_con = [0.0, 0.48], y_pred=y_pred_valid_data[:, 2].detach().numpy(), y_actual=Y_label[:, 2].numpy())
            print("------------------CONSTRAINT-PLOTS-----------------------")
            Validation.plot_unity(y_pred=torch.sum(y_pred_valid_data[:,0:2], dim=1).detach().numpy(), y_actual=torch.sum(X_data[:,0:2], dim=1).detach().numpy())
            Validation.plot_regular(x_con = [0.0, 0.48],x=X_data[:, 2].numpy(), y_act=Y_data[:, 2].numpy(), y_pred=y_pred_valid_data[:, 2].detach().numpy())
            print("------------------DONE-----------------------")

            # if True:
            #     from sklearn.metrics import r2_score
            #     print(f"Y1 R-square = {r2_score(y_pred_valid_data[:, 0].detach().numpy(), Y_label[:, 0].numpy()):.3f}")
            #     print(f"Y2 R-square = {r2_score(y_pred_valid_data[:, 1].detach().numpy(), Y_label[:, 1].numpy()):.3f}")
            #     print(f"Y3 R-square = {r2_score(y_pred_valid_data[:, 2].detach().numpy(), Y_label[:, 2].numpy()):.3f}")
            #     print(f"Equality R-square = {r2_score(torch.sum(y_pred_valid_data[:,0:2], dim=1).detach().numpy(), torch.sum(X_data[:,0:2], dim=1).detach().numpy()):.3f}")
            #     print("------------------DONE-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epoch", type=int, default=10001)
    parser.add_argument("-bs", "--batch_size", type=int, default=250)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=3)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=3)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-sh_d", "--shuffle_data", type=bool, default=True)
    parser.add_argument("-w_syn_loss", "--weight_fact_syn_data", type=float, default=[30.0, 30.0])

    args = parser.parse_args()

    train(args)