import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger():
    def __init__(self, args, data, save_dir, opt_targets):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.opt_targets = opt_targets
        self.args = args

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        self.X_train = data.X[data.train_indices].to(self.DTYPE).to(self.DEVICE)
        self.eq_cm_train = data.eq_cm[data.train_indices].to(self.DTYPE).to(self.DEVICE)
        self.ineq_cm_train = data.ineq_cm[data.train_indices].to(self.DTYPE).to(self.DEVICE)
        self.eq_rhs_train = data.eq_rhs[data.train_indices].to(self.DTYPE).to(self.DEVICE)
        self.ineq_rhs_train = data.ineq_rhs[data.train_indices].to(self.DTYPE).to(self.DEVICE)

        self.X_valid = data.X[data.valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.eq_cm_valid = data.eq_cm[data.valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.ineq_cm_valid = data.ineq_cm[data.valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.eq_rhs_valid = data.eq_rhs[data.valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.ineq_rhs_valid = data.ineq_rhs[data.valid_indices].to(self.DTYPE).to(self.DEVICE)

        if self.opt_targets:
            self.Y_target_train = data.opt_targets["y_operational"][data.train_indices].to(self.DTYPE).to(self.DEVICE)
            self.mu_target_train = data.opt_targets["mu_operational"][data.train_indices].to(self.DTYPE).to(self.DEVICE)  
            self.lamb_target_train = data.opt_targets["lamb_operational"][data.train_indices].to(self.DTYPE).to(self.DEVICE)
            self.Y_target_valid = data.opt_targets["y_operational"][data.valid_indices].to(self.DTYPE).to(self.DEVICE)
            self.mu_target_valid = data.opt_targets["mu_operational"][data.valid_indices].to(self.DTYPE).to(self.DEVICE)  
            self.lamb_target_valid = data.opt_targets["lamb_operational"][data.valid_indices].to(self.DTYPE).to(self.DEVICE)
    
    def close(self):
        self.writer.close()

    def log_loss(self, loss, network, step):

        self.writer.add_scalar(f"Train_loss/{network}_loss", loss, step)

    def log_train(self, data, primal_net, dual_net, rho, step):
        with torch.no_grad():
            Y = primal_net(self.X_train, self.eq_rhs_train, self.ineq_rhs_train)
            mu, lamb = dual_net(self.X_train, self.eq_cm_train)
            obj = data.obj_fn(Y) # Containes penalization of negative missed demand
            obj_train = data.obj_fn_train(Y) # Does not penalize negative missed demand
            dual_obj = data.dual_obj_fn(self.eq_rhs_train, self.ineq_rhs_train, mu, lamb)

            # print(Y[0, 5*14:5*15].tolist())

            ineq_resid = data.ineq_resid(Y, self.ineq_cm_train, self.ineq_rhs_train)
            ineq_dist = data.ineq_dist(Y, self.ineq_cm_train, self.ineq_rhs_train)

            eq_resid = data.eq_resid(Y, self.eq_cm_train, self.eq_rhs_train)

            if self.opt_targets:
                obj_target = data.obj_fn_log(self.Y_target_train)
                dual_obj_target = data.dual_obj_fn(self.eq_rhs_train, self.ineq_rhs_train, self.mu_target_train, self.lamb_target_train)
                # dual_obj_target = obj_target # With LP, there is strong duality, so dual obj = primal obj.
                self.writer.add_scalar(f"Train_obj/obj_optimality_gap", ((obj - obj_target)/obj_target).mean(), step)
                self.writer.add_scalar(f"Train_obj/dual_obj_optimality_gap", ((dual_obj_target - dual_obj)/dual_obj_target).mean(), step)

            # Obj funcs
            self.writer.add_scalar(f"Train_obj/obj", obj.mean(), step)
            self.writer.add_scalar(f"Train_obj/dual_obj", dual_obj.mean(), step)
            self.writer.add_scalar(f"Train_obj/duality_gap", ((obj - dual_obj)/obj).mean(), step)

            # Loss components
            # lagrange_ineq = torch.sum(mu * ineq_resid, dim=1)  # Shape (batch_size,)
            # lagrange_ineq = torch.sum(mu * ineq_resid.clamp(min=0), dim=1)  # Shape (batch_size,)
            # lagrange_eq = torch.sum(lamb * eq_resid, dim=1)   # Shape (batch_size,)
            # violation_ineq = torch.sum(torch.maximum(ineq_resid, torch.zeros_like(ineq_resid)) ** 2, dim=1)
            # violation_eq = torch.sum(eq_resid ** 2, dim=1)
            # penalty = rho/2 * (violation_ineq + violation_eq)
            # penalty = rho/2 * (Y[:, data.md_indices] ** 2)

            self.writer.add_scalar(f"Train_loss_components/obj_train", obj_train.mean(), step)
            if self.args["penalize_md_obj"]:
                lagrange_ineq = torch.sum(mu * ineq_resid, dim=1).clamp(min=0)  # Shape (batch_size,)
                violation_ineq = torch.sum(ineq_resid.clamp(min=0) ** 2, dim=1)
                penalty = rho/2 * violation_ineq
                self.writer.add_scalar(f"Train_loss_components/primal_lagrange_ineq", lagrange_ineq.mean(), step)
                self.writer.add_scalar(f"Train_loss_components/primal_penalty_term", penalty.mean(), step)
            else:
                lagrange_eq = torch.sum(lamb * Y[:, data.md_indices])
                violation_eq = torch.sum(Y[:, data.md_indices] ** 2, dim=1)
                penalty = rho/2 * violation_eq
                self.writer.add_scalar(f"Train_loss_components/primal_lagrange_eq", lagrange_eq.mean(), step)
                self.writer.add_scalar(f"Train_loss_components/primal_penalty_term", penalty.mean(), step)

            # Neural network outputs and targets
            # self.writer.add_scalar(f"Train_outputs/Y", Y.mean(), step)
            # self.writer.add_scalar(f"Train_outputs/mu", mu.mean(), step)
            # self.writer.add_scalar(f"Train_outputs/lamb", lamb.mean(), step)
            # if self.opt_targets:
            #     if data.args["benders_compact"]:
            #         Y_diff = (Y[:, data.num_g:] - self.Y_target_train).abs()
            #         lamb_diff = (lamb[:, data.num_g:] - self.lamb_target_train).abs()
            #     else:
            #         Y_diff = (Y - self.Y_target_train).abs()
            #         lamb_diff = (lamb - self.lamb_target_train).abs()
            #     mu_diff = (mu - self.mu_target_train).abs()
            #     self.writer.add_scalar(f"Train_outputs/Y_diff", Y_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_outputs/mu_diff", mu_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_outputs/lamb_diff", lamb_diff.mean(), step)

            # Constraint violations
            # self.writer.add_scalar(f"Train_constraints/eq_resid", eq_resid.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_resid", ineq_resid.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_mean", ineq_dist.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_max", ineq_dist.max(), step)
            # self.writer.add_scalar(f"Train_constraints/eq_mean", eq_resid.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraints/eq_max", eq_resid.abs().max(), step)

            p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)
            for i in range(p_gt.shape[1]):
                self.writer.add_scalar(f"Train_decvars/generator{i}", p_gt[:, i].mean(), step)
            for i in range(f_lt.shape[1]):
                self.writer.add_scalar(f"Train_decvars/lineflow{i}", f_lt[:, i].mean(), step)
            for i in range(md_nt.shape[1]):
                self.writer.add_scalar(f"Train_decvars/missed_demand{i}", md_nt[:, i].mean(), step)

            if self.opt_targets:
                # Primal variable specific differences
                
                p_gt_target, f_lt_target, md_nt_target = data.split_dec_vars_from_Y(self.Y_target_train, log=True)
                diff_p_gt = p_gt - p_gt_target
                diff_f_lt = f_lt - f_lt_target
                diff_md_nt = md_nt - md_nt_target

                net_flow = data.net_flow(f_lt)
                net_flow_target = data.net_flow(f_lt_target)
                diff_net_flow = net_flow - net_flow_target

                # diff_ui_g = (Y[:, data.ui_g_indices] - Y_target[:, data.ui_g_indices])
                self.writer.add_scalar(f"Train_var_diffs/diff_p_gt", diff_p_gt.abs().mean(), step)
                self.writer.add_scalar(f"Train_var_diffs/diff_f_lt", diff_f_lt.abs().mean(), step)
                self.writer.add_scalar(f"Train_var_diffs/diff_md_nt", diff_md_nt.abs().mean(), step)
                self.writer.add_scalar(f"Train_var_diffs/diff_net_flow", diff_net_flow.abs().mean(), step)
                # self.writer.add_scalar(f"Train_var_diffs/diff_ui_g", diff_ui_g.mean(), step)

            h, b, d, e, i, j = data.split_ineq_constraints(ineq_dist)
            ui_g, c = data.split_eq_constraints(eq_resid)

            # self.writer.add_scalar(f"Train_constraint_specific/p_gt_ub", b.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/node_balance", c.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/f_lt_lb", d.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/f_lt_ub", e.abs().mean(), step)
            # # self.writer.add_scalar(f"Train_constraint_specific/f", f.mean(), step)
            # # self.writer.add_scalar(f"Train_constraint_specific/g", g.mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/p_gt_lb", h.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/md_nt_lb", i.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/md_nt_ub", j.abs().mean(), step)
            
            # if self.opt_targets:
            #     # Dual variable specific differences
            #     # inequality
            #     mu_h, mu_b, mu_d, mu_e, mu_i, mu_j = data.split_ineq_constraints(mu)
            #     mu_target_h, mu_target_b, mu_target_d, mu_target_e, mu_target_i, mu_target_j = data.split_ineq_constraints(self.mu_target_train)
            #     mu_h_diff = mu_target_h - mu_h
            #     mu_b_diff = mu_target_b - mu_b
            #     mu_d_diff = mu_target_d - mu_d
            #     mu_e_diff = mu_target_e - mu_e
            #     mu_i_diff = mu_target_i - mu_i
            #     mu_j_diff = mu_target_j - mu_j
            #     # # equality
            #     ui_g, lamb_c = data.split_eq_constraints(lamb)
            #     ui_g, lamb_target_c = data.split_eq_constraints(self.lamb_target_train, log=True)
            #     lamb_c_diff = lamb_target_c - lamb_c

            #     self.writer.add_scalar(f"Train_dual_var_diffs/gen_ub", mu_b_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/node_balance", lamb_c_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_lb", mu_d_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_ub", mu_e_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/gen_lb", mu_h_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/md_lb", mu_i_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/md_ub", mu_j_diff.mean(), step)

            # Dual constraints
            dual_eq_resid = data.dual_eq_resid(mu, lamb, self.eq_cm_train, self.ineq_cm_train)
            dual_ineq_resid = data.dual_ineq_resid(mu, lamb)
            dual_ineq_dist = torch.clamp(dual_ineq_resid, 0)
            self.writer.add_scalar("Dual_constraints/eq_resid", dual_eq_resid.abs().mean(), step)
            self.writer.add_scalar("Dual_constraints/ineq_mean", dual_ineq_dist.mean(), step)


            # Log gradients
            # Iterate over all layers and log their gradients
            for name, param in primal_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    self.writer.add_scalar(f"Gradients_primal/{name}", param.grad.norm().item(), step)
            
            for name, param in dual_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    self.writer.add_scalar(f"Gradients_dual/{name}", param.grad.norm().item(), step)

    def log_rho_vk(self, rho, v_k, step):
        self.writer.add_scalar(f"Rho_and_violation/rho", rho, step)
        self.writer.add_scalar(f"Rho_and_violation/v_k", v_k, step)

    def log_val(self, data, primal_net, dual_net, step):
        with torch.no_grad():
            Y = primal_net(self.X_valid, self.eq_rhs_valid, self.ineq_rhs_valid)
            mu, lamb = dual_net(self.X_valid, self.eq_cm_valid)
            obj = data.obj_fn(Y) # Containes penalization of negative missed demand
            obj_train = data.obj_fn_train(Y) # Does not penalize negative missed demand
            dual_obj = data.dual_obj_fn(self.eq_rhs_valid, self.ineq_rhs_valid, mu, lamb)

            if self.opt_targets:
                # Y_target = data.opt_targets["y_operational"][data.valid_indices]
                # mu_target = data.opt_targets["mu_operational"][data.valid_indices]
                # lamb_target = data.opt_targets["lamb_operational"][data.valid_indices]
                obj_target = data.obj_fn_log(self.Y_target_valid)
                dual_obj_target = data.dual_obj_fn(self.eq_rhs_valid, self.ineq_rhs_valid, self.mu_target_valid, self.lamb_target_valid)
                self.writer.add_scalar(f"Validation/obj_optimality_gap", ((obj - obj_target)/obj_target).mean(), step)
                self.writer.add_scalar(f"Validation/dual_obj_optimality_gap", (-(dual_obj - dual_obj_target)/dual_obj_target).mean(), step)

            ineq_dist = data.ineq_dist(Y, self.ineq_cm_valid, self.ineq_rhs_valid)

            eq_resid = data.eq_resid(Y, self.eq_cm_valid, self.eq_rhs_valid)

            # Obj funcs
            self.writer.add_scalar(f"Validation/obj", obj.mean(), step)
            self.writer.add_scalar(f"Validation/dual_obj", dual_obj.mean(), step)
            # Constraint violations
            self.writer.add_scalar(f"Validation/ineq_mean", ineq_dist.mean(), step)
            self.writer.add_scalar(f"Validation/ineq_max", ineq_dist.max(), step)
            self.writer.add_scalar(f"Validation/eq_mean", eq_resid.abs().mean(), step)
            self.writer.add_scalar(f"Validation/eq_max", eq_resid.abs().max(), step)