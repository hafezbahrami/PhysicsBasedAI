import torch
torch.set_default_dtype(torch.float64)


class CVP_model(torch.nn.Module):
    """A deep learning model, with Constraint Violation Penalty"""
    def __init__(
            self, 
            x_vec_size=1, 
            y_vec_size=1, 
            layers=[20, 20],
    ):
        super().__init__()
        self.x_vec_size = x_vec_size
        self.y_vec_size = y_vec_size
        self.layers = layers

        L = []
        c = x_vec_size
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.Tanh())
            c = l
        L.append(torch.nn.Linear(c, y_vec_size))
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        return self.network(x)


class DRL_model(torch.nn.Module):
    """A deep learning model, with Data reconciliation Layer to impose equality constraint."""
    def __init__(
            self, 
            x_vec_size=1, 
            y_vec_size=1, 
            layers=[20, 20],
            constraint_coefs=None, 
            error_covar=None,
    ):
        super().__init__()
        self.x_vec_size = x_vec_size
        self.y_vec_size = y_vec_size
        self.layers = layers
        self.constraint_coefs = constraint_coefs
        self.constraint_coefs_transpose = torch.transpose(constraint_coefs, 0, 1)
        self.error_covar = error_covar  
        self.mask = torch.nn.Parameter(torch.ones(self.x_vec_size+self.y_vec_size+1), requires_grad=True)

        L = []
        c = x_vec_size
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.Tanh())
            c = l
        L.append(torch.nn.Linear(c, y_vec_size))
        L.append(torch.nn.Tanh())       
        self.network = torch.nn.Sequential(*L) 

    def forward(self, x):
        y_unconstrained = self.network(x)
        y_constrained = self.projection_matrix(x, y_unconstrained)
        return y_constrained

    def projection_matrix(self, x, y_unconstrained):
        error_covar_masked = torch.matmul(torch.diag(self.mask), self.error_covar)

        denominator = torch.inverse(
            torch.matmul(
                self.constraint_coefs, 
                torch.matmul(error_covar_masked, self.constraint_coefs_transpose)
            )
        )

        Projection = torch.eye(error_covar_masked.shape[0]) - torch.matmul(
                torch.matmul(error_covar_masked, self.constraint_coefs_transpose), 
                torch.matmul(denominator, self.constraint_coefs)
            )
        
        G_indep = Projection[self.x_vec_size:(self.x_vec_size+self.y_vec_size), 0:self.x_vec_size]
        G_indep_transpose = torch.transpose(G_indep, 0, 1)
        G_dep = Projection[self.x_vec_size:(self.x_vec_size+self.y_vec_size), self.x_vec_size:(self.x_vec_size+self.y_vec_size)]
        G_dep_transpose = torch.transpose(G_dep, 0, 1)
        Bias_proj = Projection[self.x_vec_size:(self.x_vec_size+self.y_vec_size), -1]

        y_constrained = torch.add(torch.matmul(y_unconstrained, G_dep_transpose) + torch.matmul(x, G_indep_transpose), Bias_proj)

        return y_constrained   


class _Helper:
    """This class provides all the necessary helper methods"""
    @classmethod
    def DRL_cov_matrix(
            cls,
            y_var=None,
            input_size=1,
            output_size=1,
    ):
        """
        This method will calculate the covariance matrix, which later will be defined as a learning param in the network
        """
        # Dataset co-variance matrix
        error_covar = torch.zeros(1+input_size+output_size, 1+input_size+output_size)
        for i in range(0, output_size):
            error_covar[input_size+i, input_size+i] = y_var[i]

        return error_covar#, constraint_coefs


class ConstraintLoss:
    def __init__(
        self, 
        weight_eq_loss=[0.0], 
        weight_ineq_loss=[0.0], 
        equality_constraints=None, 
        inequality_constraints=None, 
        x_vector_size=1,
        ideal_loss=None
    ):
        self.weight_eq_loss = weight_eq_loss
        self.weight_ineq_loss = weight_ineq_loss
        self.equality_constraints_X_coefs = equality_constraints[:,:x_vector_size]
        self.equality_constraints_Y_coefs = equality_constraints[:,x_vector_size:-1]
        self.equality_constraints_bias = equality_constraints[:,-1]
        self.inequality_constraints_X_coefs = inequality_constraints[:,:x_vector_size]
        self.inequality_constraints_Y_coefs = inequality_constraints[:,x_vector_size:-1]
        self.inequality_constraints_bias = inequality_constraints[:,-1]
        self.num_eq_constraints = len(equality_constraints)
        self.num_ineq_constraints = len(inequality_constraints)
        self.ideal_loss=ideal_loss

    def eq_constraint(self, x_syn=None, y_pred_syn=None, loss_syn=None):
        loss_val_eq_constraint = torch.tensor(0.)
        for i in range(self.num_eq_constraints):
            LHS = self.equality_constraints_bias[i] + torch.matmul(self.equality_constraints_X_coefs[i], torch.transpose(x_syn ,0, 1)) + torch.matmul(self.equality_constraints_Y_coefs[i], torch.transpose(y_pred_syn ,0, 1))
            loss_val_constraint = loss_syn(LHS, self.ideal_loss)
            loss_val_eq_constraint += self.weight_eq_loss[i] * loss_val_constraint
        return loss_val_eq_constraint

    def ineq_constraint(self, x_syn=None, y_pred_syn=None, loss_syn=None):
        loss_val_ineq_constraint = torch.tensor(0.)
        for i in range(self.num_ineq_constraints):
            LHS = self.inequality_constraints_bias[i] + torch.matmul(self.inequality_constraints_X_coefs[i], torch.transpose(x_syn ,0, 1)) + torch.matmul(self.inequality_constraints_Y_coefs[i], torch.transpose(y_pred_syn ,0, 1))
            sign_inequality = 0.5 * (torch.sgn(LHS) - 1.)
            pred_inequality = sign_inequality * LHS
            loss_val_constraint = loss_syn(pred_inequality, self.ideal_loss)
            loss_val_ineq_constraint += self.weight_ineq_loss[i] * loss_val_constraint
        return loss_val_ineq_constraint

    def eq_and_ineq_constraints(
            self,
            constraint_imp_approach="CVP",
            x_syn=None,
            y_pred_syn=None,
            loss_syn_torch=None,
    ):
        ineq_loss = self.ineq_constraint(x_syn=x_syn, y_pred_syn=y_pred_syn, loss_syn=loss_syn_torch)
        if constraint_imp_approach == "DRL":  # for DRL the equality constraint will be taken care of through DR layer
            return ineq_loss

        eq_loss = self.eq_constraint(x_syn=x_syn, y_pred_syn=y_pred_syn, loss_syn=loss_syn_torch)
        return eq_loss + ineq_loss


class Constraints:
    def __init__(
        self, 
        equality_constraint=None,
        inequality_constraint=None,
        data_min=None, 
        data_max=None, 
        x_vector_size=1, 
        y_vector_size=1
    ):
        self.equality_constraint = equality_constraint
        self.inequality_constraint = inequality_constraint
        self.data_min = data_min
        self.data_max = data_max
        self.x_vector_size = x_vector_size
        self.y_vector_size = y_vector_size
        
        self.upper_bound = 1.
        self.lower_bound = 0.
        self.data_min_list, self.data_max_list, self.data_idx_list = self.create_min_max_list()

    def create_min_max_list(self):
        data_min_in_list = {f"x{1+i}": self.data_min[i] for i in range(self.x_vector_size)}
        data_min_out_list = {f"y{1+i}": self.data_min[i+self.x_vector_size] for i in range(self.y_vector_size)}
        data_min_list = {**data_min_in_list, **data_min_out_list}

        data_max_in_list = {f"x{1+i}": self.data_max[i] for i in range(self.x_vector_size)}
        data_max_out_list = {f"y{1+i}": self.data_max[i+self.x_vector_size] for i in range(self.y_vector_size)}
        data_max_list = {**data_max_in_list, **data_max_out_list}

        data_in_idx_list ={f"x{1+i}": i for i in range(self.x_vector_size)}
        data_out_idx_list ={f"y{1+i}": i+self.x_vector_size for i in range(self.y_vector_size)}
        data_idx_list = {**data_in_idx_list, **data_out_idx_list, 'bias': (self.x_vector_size+self.y_vector_size)} 

        return data_min_list, data_max_list, data_idx_list
    
    def process_equality_constraints(self):
        return self._process_constraints(self.equality_constraint)
    
    def process_inequality_constraints(self):
        return self._process_constraints(self.inequality_constraint)

    def _process_constraints(self, constraint):
        counter = 0
        coefs_matrix = torch.zeros(len(constraint), 1+self.x_vector_size+self.y_vector_size)
        scale_exprs = []
        for expr in constraint:
            bias = expr[-1][0]
            scale_expr = []
            for term in expr[:-1]:
                if len(term) == 2:
                    scale_term, scale_bias = self._scale_terms_2(term, self.data_min_list, self.data_max_list)
                elif len(term) == 3:
                    scale_term, scale_bias = self._scale_terms_3(term, self.data_min_list, self.data_max_list)
                else:
                    raise ValueError("Can only handle a maximum of 2 variables per term.")
                scale_expr.extend(scale_term)
                bias += scale_bias

                idx = self.data_idx_list[scale_term[0][1]]
                coefs_matrix[counter, idx] = scale_term[0][0].item()

            scale_expr.append((bias,))
            scale_exprs.append(scale_expr)

            coefs_matrix[counter, -1] = bias.item()
            counter += 1
        return scale_exprs, coefs_matrix

    def _scale_terms_2(self, term, scale_min, scale_max):
        coef, k = term
        scale_term = [((scale_max[k] - scale_min[k]) * coef / (self.upper_bound - self.lower_bound), k,)]
        scale_bias = coef * (
            -self.lower_bound * (scale_max[k] - scale_min[k]) / (self.upper_bound - self.lower_bound)
            + scale_min[str(k)]
        )
        return scale_term, scale_bias

    def _scale_terms_3(self, term, scale_min, scale_max):
        coef, k1, k2 = term
        coef_k1 = (scale_max[k1] - scale_min[k1]) / (self.upper_bound - self.lower_bound)
        coef_k2 = (scale_max[k2] - scale_min[k2]) / (self.upper_bound - self.lower_bound)
        bias_k1 = (
            -self.lower_bound * (scale_max[k1] - scale_min[k1]) / (self.upper_bound - self.lower_bound)
            + scale_min[str(k1)]
        )
        bias_k2 = (
            -self.lower_bound * (scale_max[k2] - scale_min[k2]) / (self.upper_bound - self.lower_bound)
            + scale_min[str(k2)]
        )
        scale_term = [
            (coef * coef_k1 * coef_k2, k1, k2),
            (coef * coef_k1 * bias_k2, k1),
            (coef * coef_k2 * bias_k1, k2),
        ]
        i = 0
        while i < len(scale_term):
            if scale_term[i][0] == 0.0:
                del scale_term[i]
            else:
                i += 1
        scale_bias = coef * bias_k1 * bias_k2
        return scale_term, scale_bias