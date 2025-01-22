import copy
import torchvision
import torch


def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)  # Utilizzo di torch.numel() per calcolare la dimensione del tensore        
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size


class F_CMA(torch.optim.Optimizer):

    def __init__(self, params,
                 zeta=1e-2,
                 theta=0.5,
                 delta=0.9,
                 gamma=1e-6,
                 tau=1e-2,
                 verbose=False,
                 max_it_EDFL=100,
                 verbose_EDFL=False,
                 eta=0.5,
                 tol_zeta=1e-10,
                 momentum=0.0,
                 nesterov=False,
                 dampening=0.0):


        defaults = dict(zeta=zeta, theta=theta,
                    delta=delta,gamma=gamma,verbose=verbose,maximize=False,
                    tau=tau,max_it_EDFL=max_it_EDFL,verbose_EDFL=verbose_EDFL,
                    eta=eta,tol_zeta=tol_zeta,momentum=momentum,nesterov=nesterov,dampening=dampening)

        if momentum > 0:
            self.b = []  # Initialize momentum buffer
            for j,group in enumerate(self.param_groups):
                bb = []
                for k,p in enumerate(group['params']):
                    bb.append(None)
                self.b.append(bb)

        super().__init__(params, defaults)

    def set_zeta(self,zeta):
        for group in self.param_groups:
            group['zeta'] = zeta

    def set_fw0(self,fw0: float):
        self.fw0 = fw0

    def set_f_tilde(self,f_tilde: float):
        self.f_tilde = f_tilde

    def set_phi(self,phi: float):
        self.phi = phi

    def set_buffer_list(self,b:torch.tensor,j:int):
        self.b[0][j] = b

    def step(self, closure=None,*args,**kwargs):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure(*args,**kwargs)  #This should be used only when computing the loss on the whole data set
        else:
            with torch.no_grad():
                for j,group in enumerate(self.param_groups):
                    mu, tau = self.param_groups[j]['momentum'], self.param_groups[j]['dampening']
                    nesterov = self.param_groups[j]['nesterov']
                    for k,p in enumerate(group['params']):
                        if mu==0:
                            p.add_(p.grad, alpha=-group['zeta'])
                        else:
                            if self.b[j][k] is None:
                                buffer = p.grad.clone().detach()
                                self.set_buffer_list(buffer,k)
                                p.add_(self.b[j][k], alpha=-group['zeta'])
                            else:
                                grad = p.grad.data
                                buffer = self.b[j][k].mul_(mu).add_(grad,alpha=1-tau)
                                self.set_buffer_list(buffer,k)
                                if nesterov == False:
                                    direction = self.b[j][k]
                                else:
                                    direction = mu*self.b[j][k] + grad
                                p.add_(direction,alpha=-group['zeta'])

        return loss


    def EDFL(self,
             mod: torchvision.models,
             dl_train: torch.utils.data.DataLoader,
             w_before: torch.Tensor,
             f_tilde: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn,
             norm_d_k = None):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        delta = self.defaults.get('delta')
        eta = self.defaults.get('eta')
        verbose = self.defaults.get('verbose_EDFL')
        alpha = zeta * eta
        nfev = 0
        sample_model = copy.deepcopy(mod.to(device))
        real_loss = closure(dl_train,sample_model,criterion,device, probabilistic=False) #Compute the real loss before entering the EDFL loop

        if norm_d_k is None: norm_d_k = torch.linalg.norm(d_k)

        if verbose: print(f'Starting EDFL  with alpha =  {alpha}    f_tilde = {f_tilde}    real_loss_before = {real_loss}')
        nfev += 1
        if f_tilde > max(real_loss - gamma * alpha * norm_d_k ** 2, self.fw0):
            if verbose: print('fail: ALPHA = 0')
            alpha = 0
            return alpha, nfev, f_tilde

        w_prova = w_before + d_k * (alpha / delta)
        with torch.no_grad():
            idx = 0
            for param in sample_model.parameters():
                param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))
                idx += param.numel()

        cur_loss = closure(dl_train,sample_model,criterion,device, probabilistic=True) #Notice that here we use the approximating model \psi as described in the paper
        print(f'cur loss = {cur_loss}')
        nfev += 1

        idx = 0
        f_j = f_tilde
        
        while cur_loss <= min(f_j, real_loss - gamma * alpha * norm_d_k ** 2) and idx <= self.defaults.get('max_it_EDFL'):
            if verbose:
                print(f'idx = {idx}   cur_loss = {cur_loss}')
            f_j = cur_loss
            alpha = alpha / delta
            w_prova = w_before + d_k * (alpha / delta)
            with torch.no_grad():
                idxx = 0
                for param in sample_model.parameters():
                    param.data.copy_(w_prova[idxx:idxx + param.numel()].view(param.shape))
                    idxx += param.numel()
            cur_loss = closure(dl_train, sample_model, criterion, device, probabilistic=True) #Notice that here we use the approximating model \psi as described in the paper
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
        norm_d = torch.linalg.norm(d)
        if f_tilde < min(fw0,phi-gamma*zeta):  # This is the best case, Exit at step 7
            f_after = f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('7')
            if verbose: print('ok inner cycle')

        else:
            set_w(model,w_before)

            if verbose: print('back to w_k')

            if norm_d <= tau * zeta:  # Step 9, we check ||d||
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
                alpha, nf_EDFL, f_after_LS = self.EDFL(model,dl_train,w_before,f_tilde,
                                                       d,closure,device,criterion,norm_d)
                history['nfev'] += nf_EDFL
                if alpha * norm_d ** 2 <= tau * zeta:  # Step 14
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
                    self.set_zeta(alpha)
                    history['Exit'].append('16')

            # We set w_k+1 = w + alpha*d
            if verbose: print(f' Final alpha = {alpha}   Current step-size zeta =  {zeta}')
            if alpha > 0:  # If alpha is not zero, set the variables to the new value
                new_w = w_before + alpha * d
                set_w(model,new_w)

        if verbose: print(f'phi_before: {phi:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit Step: {history["Exit"][-1]}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        if verbose: print(f'Step-size: {self.param_groups[0]["zeta"]:3e}')
        return model, history, f_after, history['Exit'][-1]
    
