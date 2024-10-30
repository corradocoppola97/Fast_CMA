import copy
import torchvision
import torch

def get_w(model):
    weights = [p.ravel().detach() for p in model.parameters()]
    return torch.cat(weights)


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size())).item()
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size


class CMA_L(torch.optim.Optimizer):

    def __init__(self, params, alpha=1e-3, zeta=1e-3, theta=1e-3,
                 delta=1e-3,gamma=1e-3,tau=1e-2,verbose=False,max_it_EDFL=100,
                 verbose_EDFL=False):


        defaults = dict(alpha=alpha, zeta=zeta, theta=theta,
                    delta=delta,gamma=gamma,verbose=verbose,maximize=False,
                    tau=tau,max_it_EDFL=max_it_EDFL,verbose_EDFL=verbose_EDFL)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('verbose', False)
            group.setdefault('maximize', False)

    def set_zeta(self,zeta):
        for group in self.param_groups:
            group['zeta'] = zeta

    def set_fw0(self,fw0: float):
        self.fw0 = fw0

    def set_f_tilde(self,f_tilde: float):
        self.f_tilde = f_tilde

    def set_phi(self,phi: float):
        self.phi = phi

    def step(self, closure=None,*args,**kwargs):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure(*args,**kwargs)  #This should be used only when computing the loss on the whole data set
        else:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        # p.add_(torch.tensor(p.grad, dtype=torch.float), alpha=-group['zeta'])
                        p.add_(p.grad, alpha=-group['zeta'])
        return loss

    def EDFL(self,
             mod: torchvision.models,
             dl_train: torch.utils.data.DataLoader,
             w_before: torch.Tensor,
             f_tilde: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        delta = self.defaults.get('delta')
        verbose = self.defaults.get('verbose_EDFL')
        alpha = zeta
        nfev = 0
        sample_model = copy.deepcopy(mod)
        real_loss = closure(dl_train,sample_model,criterion,device)
        if verbose: print(f'Starting EDFL  with alpha =  {alpha}    f_tilde = {f_tilde}    real_loss_before = {real_loss}')
        nfev += 1
        if f_tilde > real_loss - gamma * alpha * torch.linalg.norm(d_k) ** 2:
            if verbose: print('fail: ALPHA = 0')
            alpha = 0
            return alpha, nfev, f_tilde

        w_prova = w_before + d_k * (alpha / delta)
        with torch.no_grad():
            idx = 0
            for param in sample_model.parameters():
                param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))
                idx += param.numel()

        cur_loss = closure(dl_train,sample_model,criterion,device)
        print(f'cur loss = {cur_loss}')
        nfev += 1

        idx = 0
        f_j = f_tilde
        while cur_loss <= min(f_j,real_loss - gamma * alpha * torch.linalg.norm(d_k) ** 2) and idx <= self.defaults.get('max_it_EDFL'):
            if verbose: print(f'idx = {idx}   cur_loss = {cur_loss}')
            f_j = cur_loss
            alpha = alpha / delta
            w_prova = w_before + d_k * (alpha / delta)
            with torch.no_grad():
                idxx = 0
                for param in sample_model.parameters():
                    param.copy_(w_prova[idxx:idxx + param.numel()].reshape(param.shape))
                    idxx += param.numel()
            cur_loss = closure(dl_train,sample_model,criterion,device)
            nfev += 1
            idx += 1

        return alpha, nfev, f_j


    def control_step(self,
                     model: torchvision.models,
                     w_before: torch.Tensor,
                     closure: callable,
                     dl_train: torch.utils.data.DataLoader,
                     device: torch.device,
                     criterion: torch.nn,
                     history: dict,
                     epoch: int):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        theta = self.defaults.get('theta')
        tau = self.defaults.get('tau')
        verbose = self.param_groups[0]['verbose']
        f_tilde = self.f_tilde
        fw0 = self.fw0
        phi = self.phi
        w_after = get_w(model)
        d = (w_after - w_before) / zeta  # Descent direction d_tilde

        if f_tilde < min(fw0,phi-gamma*zeta):  # This is the best case, Exit at step 7
            f_after = f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('7')
            if verbose: print('ok inner cycle')

        else:
            # Go back to the previous value and check... Maybe we can still do something. Step 8
            #model_after_IC = copy.deepcopy(model)
            #print('Loss con w after = ',closure(dl_train,model,criterion,device))
            set_w(model,w_before)
            #print('Loss con w before = ',closure(dl_train,model,criterion,device))

            if verbose: print('back to w_k')

            if torch.linalg.norm(d) <= tau * zeta:  # Step 9, we check ||d||
                if verbose: print('||d|| suff piccola  -->  Step size reduced')
                self.set_zeta(zeta * theta)  # Reduce step size, Step 10
                if f_tilde <= fw0:
                    alpha = zeta
                    new_w = w_before + alpha * d
                    set_w(model,new_w)
                    f_after = f_tilde  #Exit 10a, the tentative point is accepted  after an
                    # additional control on ||d|| but the step-size is reduced
                    history['Exit'].append('10a')
                else:
                    alpha = 0  # Exit 9b. We are no more in the level set, we cannot accept w_tilde
                    f_after = phi
                    history['Exit'].append('9b')

            else:  # Step 12, d_tilde not too small, we perform EDFL
                if verbose: print('Executing EDFL')
                #real_loss = closure(dl_train,model_after_IC,criterion,device)
                #print(f'Real Loss calcolata con model_after_IC =  {real_loss}')
                alpha, nf_EDFL, f_after_LS = self.EDFL(model,dl_train,w_before,f_tilde,
                                                       d,closure,device,criterion)
                history['nfev'] += nf_EDFL
                if alpha * torch.linalg.norm(d) ** 2 <= tau * zeta:  # Step 14
                    self.set_zeta(zeta*theta)  # Reduce the step size
                    if alpha > 0:
                        if verbose: print('LS accepted')  # Step 15a executed
                        f_after = f_after_LS
                        history['Exit'].append('15a')
                    elif alpha == 0 and f_tilde <= fw0:
                        if verbose: print('Step reduced but w_tilde accepted')  # Step 15b
                        alpha = zeta
                        f_after = f_tilde
                        history['Exit'].append('15b')
                    else:
                        if verbose: print('Total fail')  # Step 15c
                        alpha = 0
                        f_after = phi
                        history['Exit'].append('15c')
                else:  # Perform step 16, the LS is a total success, we accept alpha and do not reduce zeta
                    f_after = f_after_LS
                    history['Exit'].append('16')

            # We set w_k+1 = w + alpha*d
            if verbose: print(f' Final alpha = {alpha}   Current step-size zeta =  {zeta}')
            if alpha > 0:  # If alpha is not zero, set the variables to the new value
                new_w = w_before + alpha * d
                set_w(model,new_w)
                #print('Loss con new_w alla fine = ',closure(dl_train,model,criterion,device))

        if verbose: print(f'phi_before: {phi:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit Step: {history["Exit"][-1]}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        if verbose: print(f'Step-size: {self.param_groups[0]["zeta"]:3e}')
        return model, history, f_after, history['Exit'][-1]
    