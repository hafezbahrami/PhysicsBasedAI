import torch
import argparse
import time
from model import CVP_model, DRL_model, ConstraintLoss, _Helper, Constraints
from utils import load_data_train, rmse, preprocessing, ValidationPlot
from torchvision import transforms
torch.set_default_dtype(torch.float64)
torch.manual_seed(1373)

t0 = time.time()
def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data pre-processing to calculate the Min and Max of all train data
    data_min, data_max, data_avg, data_std = preprocessing(dataset_path="./", file_name="input_data.csv")

    min_x_val_transform = data_min[:args.x_vector_size]
    max_x_val_transform = data_max[:args.x_vector_size]
    min_x_transform = torch.cat((min_x_val_transform, min_x_val_transform), 0)
    max_x_transform = torch.cat((max_x_val_transform, max_x_val_transform), 0)
    min_y_transform = data_min[args.x_vector_size:(args.x_vector_size+args.y_vector_size)]
    max_y_transform = data_max[args.x_vector_size:(args.x_vector_size+args.y_vector_size)]
    
    train_data = load_data_train(
                dataset_path="./",
                file_name="input_data.csv",
                batch_size=args.batch_size,
                Shuffle_flag=args.shuffle_data,
                transformX=transforms.Normalize(mean=min_x_transform, std=(max_x_transform-min_x_transform)),
                transformY=transforms.Normalize(mean=min_y_transform, std=(max_y_transform-min_y_transform))
    )

    validation_data = load_data_train(
                dataset_path="./",
                file_name="validation_data.csv",
                batch_size=100000, # big batch size to get all validation in one batch for now
                Shuffle_flag=args.shuffle_data,
                transformX=None,
                transformY=None
        )
    
    # Define equality constraints in format LHS = 0. For example the following is:
    # X1 + X2 + X3 = Y1 + Y2 + Y3 ==> X1 + X2 + X3 - Y1 - Y2 - Y3 - 0.0 = 0.0
    equality_constraint = [
        [(1., 'x2'), (1., 'x4'), (-1., 'y2'), (-1., 'y3'), (-1., 'y4'), (0.,)],
        [(1., 'x1'), (1., 'x5'), (-1., 'y2'), (-1., 'y5'), (-1., 'y6'), (0.,)],
        [(-1., 'x2'), (1., 'x3'), (1., 'x5'), (-1., 'y1'), (1., 'y4'), (-1., 'y6'), (0.,)],
        [(-1., 'x3'), (1., 'x4'), (-1., 'y3'), (1., 'y5'), (-1., 'y7'), (0.,)],
    ]
    # Define inequality constraints in format LHS > 0. For example the following is:
    # Y1 < 0.9  ==> -Y1 + 0.9 > 0.0
    inequality_constraint = [
        # [(1., 'y1'), (-0.5,)],
    ]
    EqualityConstraintObj = Constraints(
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        data_min=data_min, 
        data_max=data_max, 
        x_vector_size=args.x_vector_size,
        y_vector_size=args.y_vector_size
    )
    processed_equality_constraints, equality_constraint_coefs = EqualityConstraintObj.process_equality_constraints()
    processed_inequality_constraints, inequality_constraint_coefs = EqualityConstraintObj.process_inequality_constraints()

    # Models
    if args.constraint_imp_approach == "DRL":
        error_covar = _Helper.DRL_cov_matrix(
            input_size=args.x_vector_size,
            output_size=args.y_vector_size,
            y_var=data_std[args.x_vector_size:(args.x_vector_size+args.y_vector_size)],
        )
        model = DRL_model(
            x_vec_size=args.x_vector_size,
            y_vec_size=args.y_vector_size,
            layers=args.layers,
            constraint_coefs=equality_constraint_coefs, 
            error_covar=error_covar, 
            )
        ideal_loss = torch.zeros(args.batch_size)
    elif args.constraint_imp_approach == "CVP":
        model = CVP_model(
            x_vec_size=args.x_vector_size,
            y_vec_size=args.y_vector_size,
            layers=args.layers,
            )
        ideal_loss = torch.zeros(2*args.batch_size)
    else:
        raise ValueError("Constraint implementation method must be either CVP or DRL.")
    model.to(device)

    weight_eq_loss = args.weight_equality_fact_syn_data
    weight_ineq_loss = args.weight_inequality_fact_syn_data
    ConstraintLossObj = ConstraintLoss(
        weight_eq_loss=weight_eq_loss, 
        weight_ineq_loss=weight_ineq_loss,
        equality_constraints=equality_constraint_coefs,
        inequality_constraints=inequality_constraint_coefs,
        x_vector_size=args.x_vector_size,
        ideal_loss=ideal_loss
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    # loss_syn = torch.nn.L1Loss(reduction='sum')
    # loss_data = torch.nn.L1Loss(reduction='sum')
    # optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)
    loss_syn = torch.nn.MSELoss(reduction='sum')
    loss_data = torch.nn.MSELoss(reduction='sum')

    loss_vals_epoch, loss_vals_data_epoch, loss_vals_constraint_epoch = [], [], []
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, loss_vals_data, loss_vals_constraint = [], [], []
        for X, Y in train_data:
            X, Y = X.to(device), Y.to(device)
            X_data, Y_data = X[:, :args.x_vector_size], Y

            y_pred_data = model(X_data)
            loss_val_data = loss_data(y_pred_data, Y_data)

            y_pred_syn = y_pred_data
            X_syn = X_data
            if args.constraint_imp_approach == "CVP":
                X_syn = torch.vstack((X[:, args.x_vector_size:], X_data))
                y_pred_syn = model(X_syn)

            loss_val_constraint = ConstraintLossObj.eq_and_ineq_constraints(
                constraint_imp_approach=args.constraint_imp_approach,
                x_syn=X_syn,
                y_pred_syn=y_pred_syn,
                loss_syn_torch=loss_syn,
            ) 

            loss_val = loss_val_constraint + loss_val_data
            loss_vals.append(loss_val.detach().numpy().item())
            loss_vals_data.append(loss_val_data.detach().numpy().item())
            loss_vals_constraint.append(loss_val_constraint.detach().numpy().item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        # Recording loss values
        avg_loss = sum(loss_vals) / len(loss_vals)
        loss_vals_epoch.append(avg_loss)
        avg_loss_data = sum(loss_vals_data) / len(loss_vals_data)
        loss_vals_data_epoch.append(avg_loss_data)
        avg_loss_constraint = sum(loss_vals_constraint) / len(loss_vals_constraint)
        loss_vals_constraint_epoch.append(avg_loss_constraint)

        if epoch % 100 == 0:
            print(f"Epoch = {epoch:05d}, Total loss = {avg_loss:.3f}, Data loss = {avg_loss_data:.3f}, Constraint loss = {avg_loss_constraint:.3f}")

        if epoch == args.num_epoch - 1:
            model.eval()
            verr_vals, verr_vals_data, verr_vals_constraint = [], [], []
            for X, Y_label in validation_data:
                X, Y_label = X.to(device), Y_label.to(device)
                X_data, Y_data = X[:, :args.x_vector_size], Y_label

                X_scale_var_data = (X_data - min_x_val_transform)/(max_x_val_transform - min_x_val_transform)
                
                y_pred_scale_val_data = model(X_scale_var_data)

                y_pred_val_data = min_y_transform + (max_y_transform - min_y_transform)*y_pred_scale_val_data

                # Recording error values
                verr_val_data = loss_data(y_pred_val_data, Y_data)
                verr_vals_data.append(verr_val_data.detach().numpy().item())

                ConstraintLossObj.ideal_loss = torch.zeros(args.batch_size)
                verr_val_constraint = ConstraintLossObj.eq_and_ineq_constraints(
                    constraint_imp_approach=args.constraint_imp_approach,
                    x_syn=X_data,
                    y_pred_syn=y_pred_val_data,
                    loss_syn_torch=loss_syn,
                ) 
                verr_vals_constraint.append(verr_val_constraint.detach().numpy().item())

                verr_val = verr_val_data.detach().numpy().item() + verr_val_constraint.detach().numpy().item()
                verr_vals.append(verr_val)

            t1 = time.time()
            print("-----------------RuNNiNG TiMe---------------------")
            print(f"Running time = {(t1-t0):.2f}")
            print("------------------Y-PLOTS-----------------------")
            # ValidationPlot.plot_unity_w_constraint(y_con = [], y_pred=y_pred_val_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            # ValidationPlot.plot_unity_w_constraint(y_con = [], y_pred=y_pred_val_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            # ValidationPlot.plot_unity_w_constraint(y_con = [], y_pred=y_pred_val_data[:, 2].detach().numpy(), y_actual=Y_label[:, 2].numpy())
            print("------------------CONSTRAINT-PLOTS-----------------------")
            # ValidationPlot.plot_unity_w_constraint(y_con=[0.9], y_pred=y_pred_val_data[:, 0].detach().numpy(), y_actual=Y_label[:, 0].numpy())
            # ValidationPlot.plot_unity_w_constraint(y_con=[0.3], y_pred=y_pred_val_data[:, 1].detach().numpy(), y_actual=Y_label[:, 1].numpy())
            # ValidationPlot.plot_unity(y_pred=torch.sum(y_pred_val_data, dim=1).detach().numpy(), y_actual=torch.sum(X_data, dim=1).detach().numpy())
            if True:
                import numpy as np
                ValidationPlot.plot_unity(
                        y_pred=torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,2], y_pred_val_data[:,3])), dim=0).detach().numpy(),
                        y_actual=torch.sum(torch.vstack((X_data[:,1], X_data[:,3])), dim=0).detach().numpy()
                    )
                
                ValidationPlot.plot_unity(
                        y_pred=torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,4], y_pred_val_data[:,5])), dim=0).detach().numpy(),
                        y_actual=torch.sum(torch.vstack((X_data[:,0], X_data[:,4])), dim=0).detach().numpy()
                    )
                
                ValidationPlot.plot_unity(
                        y_pred=torch.sum(torch.vstack((y_pred_val_data[:,0], -1*y_pred_val_data[:,3], y_pred_val_data[:,5])), dim=0).detach().numpy(),
                        y_actual=torch.sum(torch.vstack((-1.*X_data[:,1], X_data[:,2], X_data[:,4])), dim=0).detach().numpy()
                    )
                
                ValidationPlot.plot_unity(
                        y_pred=torch.sum(torch.vstack((y_pred_val_data[:,2], -1.*y_pred_val_data[:,4], y_pred_val_data[:,6])), dim=0).detach().numpy(),
                        y_actual=torch.sum(torch.vstack((-1.*X_data[:,2], X_data[:,3])), dim=0).detach().numpy()
                    )
                
                ValidationPlot.plot_unity(
                        y_pred=torch.sum(torch.vstack((y_pred_val_data[:,0], -1.*y_pred_val_data[:,6])), dim=0).detach().numpy(),
                        y_actual=torch.sum(torch.vstack((-1.*X_data[:,0], 2.*X_data[:,2])), dim=0).detach().numpy()
                    )
                
                print(
                        {'Constraint': 1,
                        'mean': np.mean(abs(torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,2], y_pred_val_data[:,3])), dim=0).detach().numpy() - torch.sum(torch.vstack((X_data[:,1], X_data[:,3])), dim=0).detach().numpy())), 
                        'max': np.max(abs(torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,2], y_pred_val_data[:,3])), dim=0).detach().numpy() - torch.sum(torch.vstack((X_data[:,1], X_data[:,3])), dim=0).detach().numpy()))}
                    )
                
                print(
                        {'Constraint': 2,
                        'mean': np.mean(abs(torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,4], y_pred_val_data[:,5])), dim=0).detach().numpy() - torch.sum(torch.vstack((X_data[:,0], X_data[:,4])), dim=0).detach().numpy())), 
                        'max': np.max(abs(torch.sum(torch.vstack((y_pred_val_data[:,1], y_pred_val_data[:,4], y_pred_val_data[:,5])), dim=0).detach().numpy() - torch.sum(torch.vstack((X_data[:,0], X_data[:,4])), dim=0).detach().numpy()))}
                    )
                
                print(
                        {'Constraint': 3,
                        'mean': np.mean(abs(torch.sum(torch.vstack((y_pred_val_data[:,0], -1*y_pred_val_data[:,3], y_pred_val_data[:,5])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,1], X_data[:,2], X_data[:,4])), dim=0).detach().numpy())), 
                        'max': np.max(abs(torch.sum(torch.vstack((y_pred_val_data[:,0], -1*y_pred_val_data[:,3], y_pred_val_data[:,5])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,1], X_data[:,2], X_data[:,4])), dim=0).detach().numpy()))}
                    )
                
                print(
                        {'Constraint': 4,
                        'mean': np.mean(abs(torch.sum(torch.vstack((y_pred_val_data[:,2], -1.*y_pred_val_data[:,4], y_pred_val_data[:,6])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,2], X_data[:,3])), dim=0).detach().numpy())), 
                        'max': np.max(abs(torch.sum(torch.vstack((y_pred_val_data[:,2], -1.*y_pred_val_data[:,4], y_pred_val_data[:,6])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,2], X_data[:,3])), dim=0).detach().numpy()))}
                    )
                
                print(
                        {'Constraint': 5,
                        'mean': np.mean(abs(torch.sum(torch.vstack((y_pred_val_data[:,0], -1.*y_pred_val_data[:,6])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,0], 2.*X_data[:,2])), dim=0).detach().numpy())), 
                        'max': np.max(abs(torch.sum(torch.vstack((y_pred_val_data[:,0], -1.*y_pred_val_data[:,6])), dim=0).detach().numpy() - torch.sum(torch.vstack((-1.*X_data[:,0], 2.*X_data[:,2])), dim=0).detach().numpy()))}
                    )
                
                import matplotlib.pylab as plt
                plt.figure()
                plt.plot(loss_vals_epoch)
                plt.plot(loss_vals_constraint_epoch)
                plt.plot(loss_vals_data_epoch)
                plt.yscale('log')
                plt.show()
            print("------------------DONE-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cia", "--constraint_imp_approach", type=str, default="CVP")
    parser.add_argument("-layers", "--layers", type=float, default=[25, 25])
    parser.add_argument("-n", "--num_epoch", type=int, default=20001)
    parser.add_argument("-bs", "--batch_size", type=int, default=50)
    parser.add_argument("-n_x_vec", "--x_vector_size", type=int, default=5)
    parser.add_argument("-n_y_vec", "--y_vector_size", type=int, default=7)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-6)
    parser.add_argument("-sh_d", "--shuffle_data", type=bool, default=True)
    parser.add_argument("-w_equality_syn_loss", "--weight_equality_fact_syn_data", type=float, default=[20.0, 20.0, 20.0, 20.0])
    parser.add_argument("-w_inequality_syn_loss", "--weight_inequality_fact_syn_data", type=float, default=[20.0])

    args = parser.parse_args()

    train(args)