import torch
from torch.optim.optimizer import Optimizer, required
from utils.options import args
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

device = torch.device(f"cuda:{args.gpus[0]}")


class original(object):
    """
    no ll loss
  """
    def __init__(self, model, dataloaders, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device

   
    def penalty(self, model: nn.Module):
        loss = 0
        return loss
    
    def update(self, model):
        # do nothing
        return 

def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class cdf(nn.Module):
    def __init__(self, m, s, quant_src):
        super(cdf, self).__init__()
    
        self.m = m
        self.s = s
        self.quant_src = quant_src

    def forward(self, tensor):
        normal = torch.distributions.Normal(self.m, self.s)
        cdf = normal.cdf(tensor)
        #weight_cdf = (cdf - torch.min(cdf)) / (torch.max(cdf) - torch.min(cdf)) * 2 - 1
        weight_cdf = cdf * 2 - 1
        
        if self.quant_src == 'a':
            weight_cdf = weight_cdf * args.act_range
            
        weight_pdf = torch.exp(normal.log_prob(tensor)) * 2
        return weight_cdf, weight_pdf
    
class weight_quantize_fn():
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    #assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
        
    self.uniform_q = uniform_quantize(k=self.w_bit)

  def forward(self, x):

    if self.w_bit == 32: 
      self.weight_cdf = x
      self.weight_q = x
      return x
    else:
      self.weight_cdf, self.weight_pdf = cdf(torch.mean(x), torch.std(x), 'w')(x)

      self.weight_q = self.uniform_q(self.weight_cdf)
      
      #return self.weight_q
      
      return torch.abs(self.weight_q - self.weight_cdf)/(2**self.w_bit - 1)*(2*args.act_range)*0.01
  
    
class CB_loss(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    def __init__(self, sample_num_per_cls, class_num, loss_type, beta = 0.999, gamma = 0.5):
        super(CB_loss, self).__init__()
        self.sample_num_per_cls = sample_num_per_cls
        self.class_num = class_num
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.sample_num_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.class_num
    
        labels_one_hot = F.one_hot(labels, self.class_num).float().to(device)
    
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1).to(device) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.class_num)
    
        if self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss    

class balanced_softmax_loss(nn.Module):
    def __init__(self, weight_per_class, reduction='mean'):
        super().__init__()
        self.weight_per_class = weight_per_class
        self.reduction = reduction
        
    def forward(self, logits, labels):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          sample_per_class: A int tensor of size [no of classes].
          reduction: string. One of "none", "mean", "sum"
        Returns:
          loss: A float tensor. Balanced Softmax Loss.
        """
        wpc = torch.Tensor(self.weight_per_class).type_as(logits)
        wpc = wpc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits * wpc
        loss = F.cross_entropy(input=logits, target=labels, reduction=self.reduction)
        return loss
    
class ours(object):
    def __init__(self, model, dataloaders, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        
        cls_file = os.path.join(args.csv_dir, f'reweight_{len(self.dataloaders)}.json')
        
        with open(cls_file) as json_file:
            reweight = json.load(json_file)
            
        class_num = args.num_classes
        
        weight_per_cls = [1.]*class_num
        
        for k, v in reweight.items():
            weight_per_cls[int(k)] = v
        
        self.loss = balanced_softmax_loss(weight_per_cls)
            
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract all parameters in models
        self.p_old = {} # initialize parameters
        self._precision_matrices = self._calculate_importance() 

        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old

            
  
    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items(): 
            # initialize Fisher (F) matrix（all fill zero）
            precision_matrices[n] = p.clone().detach().fill_(0)
        
      
        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num=len(self.dataloaders)
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    # get image data
                    input = data[0].to(self.device)
                    
                    # Simply use groud truth label of dataset.  
                    label = data[1].to(self.device)
                    output = self.model(input)
                    
                    feature = self.model.feature
                    fc = self.model.logit
                    
                    loss = 0.
                    
                    for _ in range(100):
                        ## simulate quantization error
                        epsilon = (torch.rand(feature.shape).to(device)*2-1)*(args.act_range)/(2**args.bitW-1)
                        noise_feature = feature + epsilon 
                        noise_output = fc(noise_feature)
                        
                        loss += -torch.log(torch.mean(F.log_softmax(noise_output, dim=1) * 
                                                      F.log_softmax(output, dim=1)))
                        
       
                    loss /= number_data*100   
                    
                    if args.balanced == 'True':
                        loss += self.loss(output, label)
                    else:
                        loss += F.nll_loss(F.log_softmax(output, dim=1), label)
                    
                    loss.backward()                                                    
                    ############################################################################

                    for n, p in self.model.named_parameters():
                        # get the gradient of each parameter and square it, then average it in all validation set.                          
                        precision_matrices[n].data += p.grad.data**2 / number_data    
                                                                             
            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] *(p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        # do nothing
        return     
    
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.from_numpy(vec)

class scp(object):
    """
    OPEN REVIEW VERSION:
    https://openreview.net/forum?id=BJge3TNKwH
    """
    def __init__(self, model: nn.Module, dataloaders: list, device, L = 50):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.p_old = {}
        self.L = L
        self.device = device
        self._precision_matrices = self.calculate_importance()
    
        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach()
    
    def calculate_importance(self):

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num = len(self.dataloaders)
            num_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    output = self.model(data[0].to(self.device))
                    
                    ####################################################################################
                    ##### generate SCP's Gamma(Γ) matrix (like MAS's Omega(Ω) and EWC's Fisher(F)) #####
                    ####################################################################################
                    #####        1.take average on a batch of Output vector to get vector φ(:,θ_A* )####
                    ####################################################################################
                    mean_vec = output.mean(dim=0)

                    ####################################################################################
                    #####   2. random sample L vectors ξ #（ Hint: sample_spherical() ）      #####
                    ####################################################################################
                    L_vectors = sample_spherical(self.L, output.shape[-1])
                    L_vectors = L_vectors.transpose(1,0).to(self.device).float()

                    ####################################################################################
                    #####   3.    每一個 vector ξ 和 vector φ( :,θ_A* )內積得到 scalar ρ               ####
                    #####           對 scalar ρ 取 backward ， 每個參數得到各自的 gradient ∇ρ           ####
                    #####       每個參數的 gradient ∇ρ 取平方 取 L 平均 得到 各個參數的 Γ scalar          ####  
                    #####              所有參數的  Γ scalar 組合而成其實就是 Γ 矩陣                      ####
                    ####(hint: 記得 每次 backward 之後 要 zero_grad 去 清 gradient, 不然 gradient會累加 )####   
                    ####################################################################################
                    total_scalar = 0
                    for vec in L_vectors:
                        scalar=torch.matmul(vec, mean_vec)
                        total_scalar += scalar
                    total_scalar /= L_vectors.shape[0] 
                    total_scalar.backward()
                    ##################################################################################      
                     
                                                
                    for n, p in self.model.named_parameters():                      
                        precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      
                        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
   
        return loss
    
    def update(self, model):
        # do nothing
        return 
    
class ewc(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
  """
    def __init__(self, model, dataloaders, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract all parameters in models
        self.p_old = {} # initialize parameters
        self._precision_matrices = self._calculate_importance() # generate Fisher (F) matrix for EWC 

        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
  
    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items(): 
            # initialize Fisher (F) matrix（all fill zero）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num=len(self.dataloaders)
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    # get image data
                    input = data[0].to(self.device)
                    # image data forward model
                    output = self.model(input)
                    # Simply use groud truth label of dataset.  
                    label = data[1].to(self.device)
                    # print(output.shape, label.shape)
                    
                    ############################################################################
                    #####                     generate Fisher(F) matrix for EWC            #####
                    ############################################################################    
                    loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
                    loss.backward()                                                    
                    ############################################################################

                    for n, p in self.model.named_parameters():
                        # get the gradient of each parameter and square it, then average it in all validation set.                          
                        precision_matrices[n].data += p.grad.data ** 2 / number_data   
                                                                            
            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # generate the final regularization term by the ewc weight (self._precision_matrices[n]) and the square of weight difference ((p - self.p_old[n]) ** 2).  
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()

        return loss
    
    def update(self, model):
        # do nothing
        return 

class mas(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """
    def __init__(self, model: nn.Module, dataloaders: list, device):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
        self.p_old = {} # initialize parameters
        self.device = device
        self._precision_matrices = self.calculate_importance() # generate Omega(Ω) matrix for MAS
    
        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
    
    def calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0) # initialize Omega(Ω) matrix（all filled zero）

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num = len(self.dataloaders)
            num_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    output = self.model(data[0].to(self.device))

                    ###########################################################################################################################################
                    #####  TODO BLOCK: generate Omega(Ω) matrix for MAS. (Hint: square of l2 norm of output vector, then backward and take its gradients  #####
                    ###########################################################################################################################################
                    output.pow_(2)                                                   
                    loss = torch.sum(output,dim=1)                                   
                    loss = loss.mean()   
                    loss.backward() 
                    ###########################################################################################################################################                          
                                            
                    for n, p in self.model.named_parameters():                      
                        precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      
                        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
   
        return loss
    
    def update(self, model):
        # do nothing
        return 

class si(object):
    """
    @inproceedings{zenke2017continual,
      title={Continual learning through synaptic intelligence},
      author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
      booktitle={International Conference on Machine Learning},
      pages={3987--3995},
      year={2017},
      organization={PMLR}
    }
        
  """
    def __init__(self, model, dataloaders, device, epsilon=0.1):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.epsilon = epsilon
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #抓出模型的所有參數
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()

    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        if self.dataloaders[0] != None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W/(p_change**2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current


                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)
        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())


        return n_p_prev, n_omega

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                _loss = omega * (p - prev_values) ** 2
                loss += _loss.sum()
         
        return loss
    
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return 
    
class rwalk(object):
    """
    @inproceedings{chaudhry2018riemannian,
      title={Riemannian walk for incremental learning: Understanding forgetting and intransigence},
      author={Chaudhry, Arslan and Dokania, Puneet K and Ajanthan, Thalaiyasingam and Torr, Philip HS},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={532--547},
      year={2018}
    }
    """
    def __init__(self, model, dataloaders, device, epsilon=0.1):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.epsilon = epsilon
        self.update_ewc_parameter = 0.4
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract model parameters and store in dictionary
        self._means = {} # initialize the guidance matrix
        self._precision_matrices = self._calculate_importance_ewc() # Generate Fisher (F) Information Matrix 
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()


    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        if self.dataloaders[0] != None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W / (1.0 / 2.0*self._precision_matrices[n] *p_change**2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = 0.5 * omega + 0.5 *omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current


                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)
        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())


        return n_p_prev, n_omega
    

    def _calculate_importance_ewc(self):
        precision_matrices = {}
        for n, p in self.params.items(): 
            n = n.replace('.', '__') # 初始化 Fisher (F) 的矩陣（都補零）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num=len(self.dataloaders)
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for n, p in self.model.named_parameters():                         
                    n = n.replace('.', '__')
                    precision_matrices[n].data *= (1 -self.update_ewc_parameter)   
                for data in dataloader:
                    self.model.zero_grad()
                    input = data[0].to(self.device)
                    output = self.model(input)
                    label = data[1].to(self.device)

                    
                    ############################################################################
                    #####                      Generate Fisher Matrix                      #####
                    ############################################################################    
                    loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
                    loss.backward()                                                    
                                                                                    
                    for n, p in self.model.named_parameters():                         
                        n = n.replace('.', '__')
                        precision_matrices[n].data += self.update_ewc_parameter*p.grad.data ** 2 / number_data  
                                                                            
            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices


    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]

                #################################################################################
                ####        Generate regularization term  _loss by omega and Fisher Matrix   ####
                #################################################################################
                _loss = (omega + self._precision_matrices[n]) * (p - prev_values) ** 2
                loss += _loss.sum()
         
        return loss
    
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return 


