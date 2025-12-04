"""
authors: (NJODE) Florian Krach & Marc Nuebel & Calypso Herrera (ONJODE Extension) Oliver Löthgren

implementation of the model for NJ-ODE
"""

# =====================================================================================================================
import torch
import numpy as np
import os
import iisignature as sig
import sklearn
import scipy.linalg
import gc

from loss_functions import LOSS_FUN_DICT


# =====================================================================================================================
def init_weights(m, bias=0.0):  # initialize weights for model for linear NN
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(bias)
def init_weights_relu(m, bias=0.0):  # initialize weights for model for linear NN
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(bias)


def save_checkpoint(model, optimizer, path, epoch, retrain_epoch=0):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model, e.g. instance of NJODE
    :param optimizer: a torch optimizer
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpt.tar')
    torch.save({'epoch': epoch,
                'weight': model.weight,
                'retrain_epoch': retrain_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               filename)


def get_ckpt_model(ckpt_path, model, optimizer, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizer: torch optimizer, which should be loaded
    :param device: the device to which the model should be loaded
    """
    ckpt_path = os.path.join(ckpt_path, 'checkpt.tar')
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt['model_state_dict']
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    model.load_state_dict(state_dict)
    model.epoch = checkpt['epoch']
    model.weight = checkpt['weight']
    if 'retrain_epoch' in checkpt:
        model.retrain_epoch = checkpt['retrain_epoch']
    if isinstance(model.readout_map, LinReg):
        model.readout_map.fitted = True
    model.to(device)



nonlinears = {  # dictionary of used non-linear activation functions. Reminder inputs
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'prelu': torch.nn.PReLU,
    'lrelu': torch.nn.LeakyReLU,
    'softmax': torch.nn.Softmax,
    'gelu': torch.nn.GELU,
}


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias, layer_norm=False):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    last_activation = None
    if isinstance(nn_desc, (tuple, list)) and len(nn_desc) > 0 \
            and isinstance(nn_desc[-1], str):
        last_activation = nn_desc[-1]
        nn_desc = nn_desc[:-1]  # remove last activation function from desc
    if nn_desc is not None and len(nn_desc) == 0:
        return torch.nn.Identity()
    if nn_desc is None or nn_desc[0] == "linear":  # if no NN desc given, or only linear
        layers = [torch.nn.Linear(in_features=input_size, out_features=output_size, bias=bias)]  # take linear NN if
        # not specified otherwise
    else:
        layers = [torch.nn.Linear(in_features=input_size, out_features=nn_desc[0][0], bias=bias)]  # first linear
        if layer_norm:
            layers.append(torch.nn.LayerNorm(nn_desc[0][0])) # layer norm
        # layer to specified dimension
        if len(nn_desc) > 1:
            for i in range(len(nn_desc) - 1):  # add multiple layers if multiple were given as input
                layers.append(nonlinears[nn_desc[i][1]]())  # add layer with specified activation function
                layers.append(torch.nn.Dropout(p=dropout_rate))  # add dropout layer
                layers.append(
                    torch.nn.Linear(nn_desc[i][0], nn_desc[i + 1][0],  # add linear layer between specified dimensions
                                    bias=bias))
                if layer_norm:
                    layers.append(torch.nn.LayerNorm(nn_desc[i + 1][0])) # layer norm
        layers.append(nonlinears[nn_desc[-1][1]]())  # last specified activation function
        layers.append(torch.nn.Dropout(p=dropout_rate))  # add another dropout layer
        layers.append(torch.nn.Linear(in_features=nn_desc[-1][0], out_features=output_size, bias=bias))  # linear
        # output layer
    if last_activation is not None:  # if a last activation function was specified, add it
        layers.append(nonlinears[last_activation]())
    return torch.nn.Sequential(*layers)  # return the constructed NN


# =====================================================================================================================
class ODEFunc(torch.nn.Module):
    """
    implementing continuous update between observatios, f_{\theta} in paper
    """

    def __init__(self, input_size, hidden_size, ode_nn, dropout_rate=0.0,
                 bias=True, input_current_t=False, input_sig=False,
                 sig_depth=3, coord_wise_tau=False, input_scaling_func="tanh",
                 use_current_y_for_ode=False, input_var_t_helper=False):
        super().__init__()  # initialize class with given parameters
        self.input_current_t = input_current_t
        self.input_sig = input_sig
        self.sig_depth = sig_depth
        self.use_current_y_for_ode = use_current_y_for_ode
        self.input_var_t_helper = input_var_t_helper
        if input_scaling_func in ["id", "identity"]:
            self.sc_fun = torch.nn.Identity()
            print("neuralODE use input scaling with identity (no scaling)")
        else:
            self.sc_fun = torch.tanh
            print("neuralODE use input scaling with tanh")

        # create feed-forward NN, f(H,X,tau,t-tau)
        if coord_wise_tau:
            add = 2*input_size
        else:
            add = 2
        if input_current_t:
            if coord_wise_tau:
                add += input_size
            else:
                add += 1
        if input_var_t_helper:
            if coord_wise_tau:
                add += input_size
            else:
                add += 1
        if input_sig:
            add += sig_depth
        if use_current_y_for_ode:
            add += input_size
        self.f = get_ffnn(  # get a feedforward NN with the given specifications
            input_size=input_size + hidden_size + add, output_size=hidden_size,
            nn_desc=ode_nn, dropout_rate=dropout_rate, bias=bias
        )

    def forward(self, x, h, tau, tdiff, signature=None, current_y=None,
                delta_t=None):
        # dimension should be (batch, input_size) for x, (batch, hidden) for h, 
        #    (batch, 1) for times

        input_f = torch.cat([self.sc_fun(x), self.sc_fun(h), tau, tdiff], dim=1)

        if self.input_current_t:
            input_f = torch.cat([input_f, tau+tdiff], dim=1)
        if self.input_var_t_helper:
            input_f = torch.cat([input_f, 1/torch.sqrt(tdiff+delta_t)], dim=1)
        if self.input_sig:
            input_f = torch.cat([input_f, signature], dim=1)
        if self.use_current_y_for_ode:
            input_f = torch.cat([input_f, self.sc_fun(current_y)], dim=1)

        df = self.f(input_f)
        return df


class GRUCell(torch.nn.Module):
    """
    Implements discrete update based on the received observations, \rho_{\theta}
    in paper
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

        self.input_size = input_size

    def forward(self, h, X_obs, i_obs):
        temp = h.clone()
        temp[i_obs] = self.gru_d(X_obs, h[i_obs])
        h = temp
        return h


class LinReg(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinReg, self).__init__()
        self.input_dim = input_size
        self.output_dim = output_size
        self.weights = torch.nn.parameter.Parameter(
            torch.empty((self.input_dim+1, self.output_dim)),
            requires_grad=False)
        self.fitted = False

    def forward(self, input):
        if not self.fitted:
            raise ValueError("LinReg has to be fitted first")
        x = torch.cat([torch.ones((input.shape[0],1)), input], dim=1)
        return torch.matmul(x, self.weights)

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
        assert y.shape[1] == self.output_dim
        for i in range(self.output_dim):
            yi = y[:, i]
            whichnan = np.isnan(yi)
            p, _, _, _ = scipy.linalg.lstsq(X[~whichnan], yi[~whichnan])
            self.weights.data[:, i] = torch.Tensor(p.squeeze())
        self.fitted = True


class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with tanh applied to inputs and the
    option to use a residual NN version
    """

    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=False, masked=False, recurrent=False,
                 input_sig=False, sig_depth=3, clamp=None, input_t=False,
                 t_size=None, nb_outputs=None,
                 **kwargs):
        super().__init__()

        self.use_lstm = False
        if nn_desc is not None and isinstance(nn_desc[0][0], str) \
                and nn_desc[0][0].lower() == "lstm":
            self.use_lstm = True
            print("USE LSTM")
        in_size = input_size
        if masked:
            in_size = 2 * input_size
        if recurrent and not self.use_lstm:
            in_size += output_size
        if input_sig:
            in_size += sig_depth
        self.masked = masked
        self.recurrent = recurrent
        self.output_size = output_size
        self.nb_outputs = nb_outputs
        if self.nb_outputs is None:
            self.nb_outputs = 1
        self.input_sig = input_sig
        self.sig_depth = sig_depth
        self.input_t = input_t
        if self.input_t:
            in_size += t_size
        self.clamp = clamp
        self.lstm = None
        if self.use_lstm:
            self.lstm = torch.nn.LSTMCell(
                input_size=in_size, hidden_size=nn_desc[0][1], bias=bias)
            self.c_h = None
            in_size = nn_desc[0][1]*2
            assert in_size == output_size, \
                "when using an LSTM, the hidden_size has to be 2* " \
                "the LSTM output size"
            nn_desc = nn_desc[1:]
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=self.output_size*self.nb_outputs,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias)

        if residual:
            print('use residual network: input_size={}, output_size={}'.format(
                input_size, output_size))
            if input_size <= output_size:
                self.case = 1
            if input_size > output_size:
                self.case = 2
        else:
            self.case = 0

    def forward(self, nn_input, mask=None, sig=None, h=None, t=None):
        identity = None
        if self.case == 1:
            identity = torch.zeros((nn_input.shape[0], self.output_size)).to(
                self.device)
            identity[:, 0:nn_input.shape[1]] = nn_input
        elif self.case == 2:
            identity = nn_input[:, 0:self.output_size]

        # add input options here in nn forward pass with options to accommodate for phi1 and phi2
        if self.recurrent or self.use_lstm:
            assert h is not None
            # x = torch.tanh(nn_input)
            x = nn_input
        else:
            x = torch.tanh(nn_input)  # maybe not helpful
        if self.recurrent and not self.use_lstm:
            x = torch.cat((x, h), dim=1)
        if self.input_t:
            x = torch.cat((x, t), dim=1)
        if self.masked:
            assert mask is not None
            x = torch.cat((x, mask), dim=1)
        if self.input_sig:
            assert sig is not None
            x = torch.cat((x, sig), dim=1)
        if self.use_lstm:
            h_, c_ = torch.chunk(h, chunks=2, dim=1)
            h_, c_ = self.lstm(x.float(), (h_, c_))
            x = torch.concat((h_, c_), dim=1)
        out = self.ffnn(x.float())

        if self.nb_outputs > 1:
            out = out.reshape(-1, self.output_size, self.nb_outputs)
            if self.case > 0:
                identity = identity.reshape(-1, self.output_size, 1).repeat(
                    1, 1, self.nb_outputs)

        if self.case == 0:
            pass
        else:
            out = identity + out

        if self.clamp is not None:
            out = torch.clamp(out, min=-self.clamp, max=self.clamp)

        return out

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

class RBFGeneralizedKernel(torch.nn.Module): # RBF weighted instead of uniform weights
    """
    ABSOLUTE MINIMAL change to existing GeneralizedKernel.
    
    Just adds RBF weights in place of uniform (1/J_i) weights.
    Uses the same phi1/phi2 structure as original.
    
    USE THIS FIRST to test if distance-based weighting helps at all.
    """
    
    def __init__(self, input_size, context_size, space_dimension, nn_desc,
                 dropout_rate=0.0, use_local=True, bias=True, masked=False,
                 initial_bandwidth=1.0, learnable_bandwidth=True, layer_norm=False):
        super().__init__()
        
        self.input_size = input_size
        self.context_size = context_size
        self.space_dimension = space_dimension
        self.use_local = use_local
        self.masked = masked
        self.layer_norm = layer_norm
        
        assert use_local, "This implementation requires use_local=True"
        
        # Same phi1 and phi2 as original
        in_size = input_size
        if self.masked:
            in_size = 2 * input_size
        phi1_input_size = in_size + 2 * self.space_dimension + 1
        phi2_input_size = 1
        
        self.phi1 = get_ffnn(
            input_size=phi1_input_size,
            output_size=self.context_size,
            nn_desc=nn_desc,
            dropout_rate=dropout_rate,
            bias=bias,
            layer_norm=self.layer_norm 
        )
        
        self.phi2 = get_ffnn(
            input_size=phi2_input_size,
            output_size=context_size,
            nn_desc=nn_desc if nn_desc else [[context_size, "tanh"]],
            dropout_rate=dropout_rate,
            bias=bias,
            layer_norm=self.layer_norm
        )
        
        # RBF bandwidth
        if learnable_bandwidth:
            self.log_sigma = torch.nn.Parameter(torch.tensor(np.log(initial_bandwidth)))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(initial_bandwidth)))
      
    
    def forward(self, xi, eval_ptr, X_obs_impute_i, space_points_i, n_space_i, M_obs=None):
        """Same as original but with RBF weights instead of 1/J_i"""
        nr_eval_points = len(xi)
        nr_batch_paths = len(n_space_i)
        c = torch.zeros(nr_eval_points, self.context_size, device=self.device)
        
        sigma = torch.exp(self.log_sigma)
        
        obs_batch_ptr = torch.cumsum(torch.cat((torch.tensor([0], device=self.device), n_space_i)), dim=0)
        
        for bp in range(nr_batch_paths):
            # eval points to update
            start_eval, end_eval = eval_ptr[bp].item(), eval_ptr[bp+1].item()
            nr_eval_points_bp = end_eval - start_eval
            xi_b = xi[start_eval:end_eval, :] # eval points for this batch path
            # data observed at this time for this batch path
            start, end = obs_batch_ptr[bp].item(), obs_batch_ptr[bp+1].item()
            J_i_b = n_space_i[bp].item() #  [nr_obs_points, ]
            X_obs_b = X_obs_impute_i[start:end, :] # [nr_obs_points, input_size]
            space_points_b = space_points_i[start:end, :] # [nr_obs_points, space_dimension(1)]
            
            if self.masked:
                M_obs_batch = M_obs[start:end, :] # same as X_obs_i
            # reshape for vectorization 
            xi_b = xi_b.unsqueeze(1) # [nr_eval_points_bp, 1, spacedim]
            xi_j_b = space_points_b.unsqueeze(0) # [1, J_i_b, spacedim]
            X_j_b = X_obs_b.unsqueeze(0) # [1, J_i_b, dim]
            if self.masked:
                M_j_b = M_obs_batch.unsqueeze(0)  # [1, J_i_b, dim]
            J_b = torch.full((nr_eval_points_bp, J_i_b, 1), J_i_b, device=self.device)  # J_i repeated for each j
            j_b = torch.arange(1, J_i_b+1, device=self.device, dtype=torch.float32).view(1, -1, 1).expand(nr_eval_points_bp, -1, 1) # indices j=1,...,J_i

            # phi1 inputs
            x1 = torch.cat((xi_b.expand(-1, J_i_b, -1), 
                            xi_j_b.expand(nr_eval_points_bp, -1, -1), 
                            X_j_b.expand(nr_eval_points_bp, -1, -1)), dim=2)
            if self.masked:
                x1 = torch.cat((x1, M_j_b.expand(nr_eval_points_bp, -1, -1)), dim=2)
            x1 = torch.cat((x1, J_b), dim=2)  # add J_i to each row
            x1 = x1.contiguous()
            
            # phi2 inputs
            x2 = j_b
            x2 = x2.contiguous()

            # compute the outputs
            phi1_out = self.phi1(x1.view(-1, x1.size(-1))).view(nr_eval_points_bp, J_i_b, self.context_size)  # [nr_eval_points_bp, J_i_b, context_size] --> first dim: eval points, second dim: spatial index
            phi2_out = self.phi2(x2.view(-1, x2.size(-1))).view(nr_eval_points_bp, J_i_b, self.context_size)
            combined = phi1_out * phi2_out  # [nr_eval_points_bp, J_i_b, context_size]
            
            # Compute distances
            delta_xi = xi_b - xi_j_b  # [nr_eval_points_bp, J_i_b, space_dim]
            distances_sq = torch.sum(delta_xi ** 2, dim=2)  # [nr_eval_points_bp, J_i_b]
            
            # RBF weights
            rbf_weights = torch.exp(- distances_sq / (2 * sigma ** 2))  # [nr_eval_points_bp, J_i_b]
            
            # Normalize per query
            weight_sums = torch.sum(rbf_weights, dim=1, keepdim=True)  # [nr_eval_points_bp, 1]
            rbf_weights = rbf_weights / (weight_sums + 1e-8)  # [nr_eval_points_bp, J_i_b]
            
            # Apply weights: [nr_eval_points_bp, J_i_b, 1] * [nr_eval_points_bp, J_i_b, context_size]
            weighted_combined = rbf_weights.unsqueeze(2) * combined
            
            # Sum over observations (instead of average)
            c[start_eval:end_eval, :] = torch.sum(weighted_combined, dim=1)
        
        return c
    
    @property
    def device(self):
        return next(self.parameters()).device


# ADDED FOR GENERALIZED KERNEL
class GeneralizedKernel(torch.nn.Module):
    """
    Generalized kernel network ψθ3(t, ξ, O_κ(t)) for ONJODE
    
    This implements equation (5) from the paper:
    ψθ3(t, ξ, O_κ(t)) = (1/J_κ(t)) * Σ_{j=1}^{J_κ(t)} φ1_θ3(ξ, ξ_j^κ(t), M_j^κ(t), X̃_t^j, J_κ(t)) ⊙ φ2_θ3(j)
    
    For initial implementation, we use a simpler general FFNN that takes (t, ξ, O_t) as input

    BUT without masking
    """
    
    def __init__(self, input_size, context_size, space_dimension, nn_desc, 
                 dropout_rate=0.0, use_local=True, bias=True, masked=False, layer_norm=False):
        """
        :param input_size: int, dimension of the X coordinates considered as input
        :param context_size: int, dimension of the context vector and output of the generalized kernel
        :param space_dimension: int, dimension of the spatial domain
        :param nn_desc: list of lists, defining the NN structure
        :param dropout_rate: float, dropout rate
        :param bias: bool, whether to use bias
        :param use_local: bool, if True use local aggregation with phi1 and phi2
        :param masked: bool, whether to use masking information
        :param layer_norm: bool, whether to use layer normalization in phi networks
        """
        super().__init__()
        
        self.input_size = input_size
        self.context_size = context_size
        self.space_dimension = space_dimension
        self.use_local = use_local
        self.masked = masked
        self.layer_norm = layer_norm

        if self.use_local:
            # Local aggregation approach with phi1 and phi2
            # phi1: processes local spatial information
            # Input: (ξ, ξ_j, M_j, X̃_j, J) 
            # Dimensions: space_dim + space_dim + input_size + input_size + 1
            in_size = input_size
            if self.masked:
                in_size = 2 * input_size
            phi1_input_size = in_size + 2 * self.space_dimension + 1
            
            # phi2: processes observation index information
            # To learn the projection based on index j
            phi2_input_size = 1  

            self.phi1 = get_ffnn(
                input_size=phi1_input_size, 
                output_size=self.context_size,
                nn_desc=nn_desc,
                dropout_rate=dropout_rate,
                bias=bias, layer_norm=self.layer_norm 
            )
            
            self.phi2 = get_ffnn(
                input_size=phi2_input_size,
                output_size=context_size,
                nn_desc=nn_desc,
                dropout_rate=dropout_rate,
                bias=bias, layer_norm=self.layer_norm 
            )

        else: 
            raise NotImplementedError("Only use_local=True is implemented in this version.")
        
    def forward(self, xi, eval_ptr, X_obs_impute_i, space_points_i, n_space_i, M_obs=None):
        """
        Forward pass through the generalized kernel
        
        :param xi: torch.tensor, spatial evaluation point of the ONJODE, [nr_obs_hidden_states + others points in observed paths, 1]
        :eval_ptr: torch.tensor, pointer to separate batches in xi, [nr_batches + 1, ]
        :param X_obs_impute_i: torch.tensor, X observations at time t_i across all batches (X_{t_i, xi_j^i}), [nr_obs_hidden_states, dim]
        :param space_points_i: torch.tensor, spatial points of observations across batches at time t_i (xi_j^i), [nr_obs_hidden_states, space_dimension(1)]
        :param n_space_i: torch.tensor, number of spatial observations across batches at time t_i (J_i), [nr_batches, ]
        :param M_obs: torch.tensor, mask for observations across batches for time t_i (M_j^i), [nr_obs_hidden_states, dim]
        :return: torch.tensor, context vector [batch_size, output_size]
        """

        if self.use_local: 
            '''
            At obs time ti, it takes fixed dimensional inputs for phi1 and phi2, 
            sums together (for each spatial point j=1, ..., J_i) the dot product between outputs of dimension context size,
            on a batch by batch basis --> updates the hidden state for observed batches at time ti
            '''
            # nr of evaluation points (to calculate hidden state for) and paths in batch
            nr_eval_points = len(xi)
            nr_batch_paths = len(n_space_i)
            c = torch.zeros(nr_eval_points, self.context_size).to(self.device)

            # restructure data so that it is batchwise of dim (obs_batch_size, data)
            # for phi1: (xi, xi_j, M_j, X̃_j, J) for j=1,...,J_i
            # for phi2: j for j=1,...,J_i
            # At this time, for each batch, the below computes the context for each evaluation point of the batch, 
            # relative to the observation points of the batch
            # This is to be vectorized over evaluation points and observation points for efficiency
            obs_batch_ptr = torch.cumsum(torch.cat((torch.tensor([0]), n_space_i.cpu())), dim=0)
            for bp in range(nr_batch_paths):
                # get relevant points from the batch for xi, X, xi_j, M_j
                # eval points to update
                start_eval, end_eval = eval_ptr[bp].item(), eval_ptr[bp+1].item()
                nr_eval_points_bp = end_eval - start_eval
                xi_b = xi[start_eval:end_eval, :] 
                # data observed at this time for this batch path
                start, end = obs_batch_ptr[bp].item(), obs_batch_ptr[bp+1].item()
                J_i_b = n_space_i[bp].item() # [nr_obs_points, ]
                X_obs_b = X_obs_impute_i[start:end, :] # [nr_obs_points, input_size]
                space_points_b = space_points_i[start:end, :] # [nr_obs_points, space_dimension(1)]
                
                if self.masked:
                    M_obs_batch = M_obs[start:end, :] # same as X_obs_i
                # reshape for vectorization
                # for phi1
                xi_b = xi_b.unsqueeze(1)  # [nr_eval_points_bp, 1, spacedim]
                xi_j_b = space_points_b.unsqueeze(0)  # [1, J_i_b, spacedim]
                X_j_b = X_obs_b.unsqueeze(0)  # [1, J_i_b, dim]
                if self.masked:
                    M_j_b = M_obs_batch.unsqueeze(0)  # [1, J_i_b, dim]
                J_b = torch.full((nr_eval_points_bp, J_i_b, 1), J_i_b, device=self.device)  # J_i repeated for each j
                # for phi2
                j_b = torch.arange(1, J_i_b+1, device=self.device, dtype=torch.float32).view(1, -1, 1).expand(nr_eval_points_bp, -1, 1) # indices j=1,...,J_i
                # phi1 inputs
                x1 = torch.cat((xi_b.expand(-1, J_i_b, -1), 
                                xi_j_b.expand(nr_eval_points_bp, -1, -1), 
                                X_j_b.expand(nr_eval_points_bp, -1, -1)), dim=2)
                if self.masked:
                    x1 = torch.cat((x1, M_j_b.expand(nr_eval_points_bp, -1, -1)), dim=2)
                x1 = torch.cat((x1, J_b), dim=2)  # add J_i to each row
                x1 = x1.contiguous()
                
                # phi2 inputs
                x2 = j_b
                x2 = x2.contiguous()
                
                # compute the outputs                
                phi1_out = self.phi1(x1.view(-1, x1.size(-1))).view(nr_eval_points_bp, J_i_b, self.context_size)  # [nr_eval_points_bp, J_i_b, context_size] eval points, obs point, context
                phi2_out = self.phi2(x2.view(-1, x2.size(-1))).view(nr_eval_points_bp, J_i_b, self.context_size)  # [nr_eval_points_bp, J_i_b, context_size]
                
                # combine outputs
                combined = phi1_out * phi2_out  # element-wise product, [nr_eval_points_bp, J_i_b, context_size]
                c[start_eval:end_eval, :] = torch.sum(combined, dim=1) / J_i_b # dim [nr_eval_points_bp, context_size]
        else:
            raise NotImplementedError("General FFNN approach for generalized kernel not implemented yet")

        return c # a [obs_batch_size, context_size] tensor of newly calculated context vectors at observation time ti, for the observed batches

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

class Operator_FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks

    Used to define:
        - evolution network (ode_f): recurrent nn with input (prev h, t, tau(t), xi) --> output h
        - jump network (encoder_map): recurrent nn with input (prev h, t, xi, genk output) --> output h
        - readout network (readout_map) in ONJODE: ffnn with input (h) --> output (Xhat prediction)
    """

    def __init__(self, nn_input_size, nn_output_size, nn_desc, dropout_rate=0.0, #
                 bias=True, residual=False, clamp=None, input_t=False,
                 eval_point=True, data = False, input_scaling_func="id",
                 t_size=None, space_dimension=None, context_size=None, layer_norm=False, input_norm=False,
                 **kwargs):
        super().__init__()
        '''
        :param nn_input_size: int, base input size, always the hidden_size. Sufficient for readout_map, with additional inputs added for ode_f and encoder_map
        :param nn_output_size: int, output size of the network. for ode_f and encoder_map, output_size=hidden_size. for readout_map, output_size=dim
        :param nn_desc: list of lists, defining the NN structure
        :param residual: bool, implement residual NNs
        :param input_t: bool, include time input t_i (or t and tau(t)). for ode_f and encoder_map
        :param eval_point: bool, include evaluation point xi as input. for ode_f and encoder_map
        :param data: bool, include genk output as input. for encoder_map
        :param input_scaling_func: str, "id" or "tanh" to scale Neural ODE inputs by tanh function
        :param t_size: int, size of time input (1 for ode_f, 2 for encoder_map)
        :param space_dimension: int, dimension of Xi (the function domain)
        :param context_size: int, size of genk output (only for encoder_map)
        :param layer_norm: bool, whether to use layer normalization in the NN
        :param input_norm: bool, whether to use layer normalization for encoder input
        '''
        # // adaptable input size for ode_f, encoder_map, readout_map
        in_size = nn_input_size # previous hidden state
        self.input_t = input_t
        if self.input_t: # When True, adds either t_i or t,tau(t) -- node & encoder
            in_size += t_size
        self.eval_point = eval_point 
        if self.eval_point: # When True, adds evaluation point xi  -- node & encoder
            in_size += space_dimension  # add spatial evaluation point dimensions
        self.data = data
        if self.data: # When True, adds generalized kernel output -- encoder
            in_size += context_size
        # in_size = hidden_size + (t_size=either t_i or t,tau(t)) + (space_dimension) + {genk output}
        self.output_size = nn_output_size
        self.clamp = clamp
        self.layer_norm = layer_norm
        self.input_norm = input_norm

        if input_scaling_func in ["id", "identity"]:
            self.sc_fun = torch.nn.Identity()
            print("neuralODE use input scaling with identity (no scaling)")
        else:
            self.sc_fun = torch.tanh
            print("neuralODE use input scaling with tanh")
        
        # check cases for residual nn
        self.case = 0
        if in_size == nn_input_size: # + space_dimension: # readout nn case TODO: added eval point with H in readout
            self.case = 3
            print("created readout nn with input size ", in_size)
        elif in_size == nn_input_size + 1 + space_dimension + 1 + context_size: # ode nn case -- add context
            self.case = 1.5
            print("created ode nn (incl. data) input size ", in_size)
        elif in_size == nn_input_size + 1 + space_dimension + 1: # ode nn case -- no context
            self.case = 1
            print("created ode nn (no data) with input size ", in_size)
        elif in_size == nn_input_size + 1 + space_dimension + context_size: # encoder nn case
            self.case = 2
            print("created encoder nn with input size ", in_size)
            if self.input_norm:
                self.in_norm = torch.nn.LayerNorm(in_size) # layer norm for encoder input
        else:
            raise ValueError("input and output sizes of the NN do not match ode_f, encoder_map, or readout_map")
        
        # // defining neural network
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=self.output_size, 
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias, layer_norm=self.layer_norm)
        
        # // residual nn option
        self.residual = residual
        if self.residual:
            print('use residual network: input_size={}, output_size={}'.format(
                in_size, nn_output_size))
            # define a projection from input to output dimensions
            self.project = torch.nn.Linear(in_size, nn_output_size)

    def forward(self, h, t=None, xi=None, c=None): 
        # Input dimensions:
        #     [nr_hidden_states, data] for ode 
        #     [nr_observed_batches, data] for encoder
        #     [nr_hidden_states, data] for readout
        # h is the default input: previous hidden state
        # t is the time input: t_i or (t_i, tau(t_i))
        # xi is the eval point
        # c is the genk context output at time t_i

        if self.case == 1: # ode without context (data)
            x = torch.cat((self.sc_fun(h), t, xi), dim=1)
        elif self.case == 1.5: # ode with context (data)
            x = torch.cat((self.sc_fun(h), t, xi, c), dim=1)
        elif self.case == 2: # encoder
            x = torch.cat((h, t, xi, c), dim=1)
            if self.input_norm:
                x = self.in_norm(x)
        elif self.case == 3: # readout
            x = h
        else:
            raise ValueError("case ode, encoder, or readout not recognized")

        out = self.ffnn(x)

        if self.residual:
            print("-----using residual connection in OPERATOR_FFNN-----")
            identity = self.project(x)
            out = identity + out
        else:
            pass

        if self.clamp is not None:
            out = torch.clamp(out, min=-self.clamp, max=self.clamp)

        return out

    @property
    def device(self):
        device = next(self.parameters()).device
        return device


class ONJODE(torch.nn.Module):
    """
    ONJODE test: models directly conditional expectation 
    """
    def __init__(self, input_size, hidden_size, context_size, output_size, space_dimension,
                 ode_nn, readout_nn, enc_nn, genk_nn, 
                 bias=True, dropout_rate=0, solver="euler",
                 weight=0.5, weight_decay=1., input_coords=None, output_coords=None, **options):
        '''
        :param input_size: int, the size of the considered input (number of input coords - else dimension)
        :param hidden_size: int, the size of the hidden state (Neural ODE and jump network outputs)
        :param context_size: int, size of the context vector from the generalized kernel
        :param output_size: int, the size of the considered output (number of output coords - else dimension)
        :param space_dimension: int, the dimension of the spatial (function) domain of the S

        :param ode_nn: list of lists, defining the node f, see get_ffnn
        :param readout_nn: list of lists, defining the readout g, see get_ffnn and nn_desc
        :param enc_nn: list of lists, defining the encoder rho, see get_ffnn and nn_desc
        :param genk_nn: list of lists, defining the generalised kernel (phi1 and phi2), see get_ffnn and nn_desc
        
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param input_coords: list of int, the coordinates of the input to consider (out of range(dimension))
        :param output_coords: list of int, the coordinates of the output to consider (out of range(dimension))
        :param options: kwargs, used:
                        - "which_loss": str, the loss function to use (default is 'operator' and this is the only supported loss for now)
                        - "residual_dec": bool, whether to use residual nn for ode_f
                        - "residual_enc": bool, whether to use residual nn for encoder_map
                        - "residual": bool, whether to use residual nn for both ode_f and encoder_map
                        - "input_scaling_func": str, the input scaling function to use ("id" for identity, else tanh)

        '''
        super().__init__()
        options1 = options['options'] if 'options' in options else {}
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.masked = options1['masked'] if 'masked' in options else False
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.space_dimension = space_dimension
        ode_t_size = 2  # for ode nn, input t has both t_i and tau(t_i)
        enc_t_size = 1  # for encoder nn, input t has only t_i

        self.solver = solver
        self.epoch = 1
        self.weight = weight
        self.weight_decay = weight_decay

        # options: input scaling, layer norms, residual NNs, clamping, local or general genk (default local)
        self.input_scaling_func = options1['input_scaling_func'] if 'input_scaling_func' in options1 else "id"
        self.layer_norm = options1['layer_norm'] if 'layer_norm' in options1 else False
        self.enc_input_norm = options1['enc_input_norm'] if 'enc_input_norm' in options1 else False
        self.residual_dec = False
        self.residual_enc = False
        self.residual_ode = False
        if 'residual_dec' in options:
            self.residual_dec = options['residual_dec']
        if 'residual_enc' in options:
            self.residual_enc = options['residual_enc']
        if 'residual_ode' in options:
            self.residual_ode = options['residual_ode']
        if 'residual' in options:
            residual = options['residual']
            self.residual_dec = residual
            self.residual_enc = residual
            self.residual_ode = residual
        self.clamp = None
        if 'clamp' in options:
            self.clamp = options['clamp']
        
        #// defining the four main neural networks
        self.ode_f = Operator_FFNN(nn_input_size=self.hidden_size, nn_output_size=self.hidden_size, nn_desc=ode_nn, dropout_rate=dropout_rate, 
                                   bias=bias, residual=self.residual_ode, clamp=self.clamp, input_t=True, eval_point=True, data=False, 
                                   t_size=ode_t_size, space_dimension=self.space_dimension, context_size=self.context_size, 
                                   input_scaling_func=self.input_scaling_func, layer_norm=self.layer_norm) # added data input to ode
        self.gen_kernel = GeneralizedKernel(input_size=self.input_size, context_size=self.context_size, space_dimension=self.space_dimension, nn_desc=genk_nn, 
                                            dropout_rate=0.0, use_local=True, bias=bias, masked=self.masked, layer_norm=self.layer_norm)
        self.encoder_map = Operator_FFNN(nn_input_size=self.hidden_size, nn_output_size=self.hidden_size, nn_desc=enc_nn, dropout_rate=dropout_rate, 
                                         bias=bias, residual=self.residual_enc, clamp=self.clamp, input_t=True, eval_point=True, data=True, 
                                         t_size=enc_t_size, space_dimension=self.space_dimension, context_size=self.context_size,
                                         layer_norm = self.layer_norm, input_norm=self.enc_input_norm) 
        self.readout_map = Operator_FFNN(nn_input_size=self.hidden_size, nn_output_size=self.output_size, nn_desc=readout_nn, dropout_rate=dropout_rate, 
                                         bias=bias, residual=self.residual_dec, clamp=self.clamp, input_t=False, eval_point=False, data=False, 
                                         t_size=None, space_dimension=self.space_dimension, context_size=None, layer_norm=False) # TODO: testing eval_point xi input with h for readout
        self.apply(init_weights) # currently Xavier initialization. Optionally use init_weights_relu for He initialization

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, current_time, xi, tau, delta_t, c=None):
        # create the time input for ode_f
        # h is dim [nr_hidden_states, hidden_size]
        # current_time is float scalar
        # xi is dim [nr_hidden_states, space_dimension]
        # tau is dim [nr_hidden_states, 1]
        # delta_t is float scalar
        t_tensor = torch.tensor([[current_time]], dtype=torch.float32, device=self.device).expand(h.shape[0], 1)
        xi = xi.unsqueeze(1) if len(xi.shape) == 1 else xi  # ensure correct shape
        time_input = torch.cat((t_tensor, tau), dim=1) # dimension: [nr_hidden_states, 2]
        
        """Executes a single ODE step"""
        if c is None:
            if self.solver == "euler":
                h = h + delta_t * self.ode_f(h=h, t=time_input, xi=xi)
            else:
                raise ValueError("Unknown solver '{}'.".format(self.solver))
        else:
            if self.solver == "euler":
                h = h + delta_t * self.ode_f(h=h, c=c, t=time_input, xi=xi)
            else:
                raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def forward(self, eval_points, eval_ptr, times, time_ptr, X, obs_idx, 
                space_points, n_space_ot, batch_ptr,
                delta_t, T, n_obs_ot, 
                evaluate=False, plot=False, paths_to_plot=(0,), space_coords=None, until_T=False, M=None,
                which_loss='operator', dim_to=None):
        '''
        forward pass through the ONJODE model to get H_T and the loss, optionally calculating an evaluation metric (mean square error across the grid) and
        plotting results.

        This requires the evaluation points xi for any hidden state that will be required later in calculation, i.e. in computing the loss, evaluation metric,
        or for plotting. All hidden states for these points must be evolved.

        :param eval_points: floats, all the evaluation points (obs space points) across time for all batches
                            dimension: [sum_b J^(b)] where J^(b) = sum_i J_i^(b), the total nr of space points over time
                                                                                  for batch b.
        :param eval_ptr: int, ptr to relevant eval_points for each batch. This is the vector [0, [J^(1), ..., J^(B)]]
        :param times: floats, 1d array of observation times
                      dimension: [count of t_i across batches, ] 
        :param time_ptr: ints, 1d array of integers pointing to indices of X corresponding to t_i in times
                         dimension: [len(times) + 1, ]
        :param X: floats, all observations X at each time, across batches and observed points in space.
                  dimension: [nr_X_obs_across_batches in time and space, dim] (see data_utils.OperatorCollateFnGen)
        :param obs_idx: ints, 1d array of batch observation indices for each group over space in time
                  dimension: [nr_obs_batches across time, ]
        :param space_points: floats, 1d array of spatial points for each observation X_i,j (1 for all dims, not repeated)
                  dimension: [nr_X_obs_across_batches in time and space, ]
        :param n_space_ot: int, array of nr of observations in space J_i^(b) at each time for each batch
                            dimension: [nr_obs_batches_through_time, time_steps]
        :param batch_ptr: int, 1d array of integers pointing to which batches are included at time t_i
                          dimension: [len(times) + 1, ]
        :param delta_t: float, time step size
        :param T: float, final time
        :param n_obs_ot: int, array of number of additional obs per batch 
                         dimension: [batch_size,]
        :param evaluate: bool, whether the call is for the purpose of evaluating the msq between the model and optimal cond. exp.
                               True: sets return_path=True, space_coords=entire grid (expected), until_T=True and replaces the unique eval 
                                     points of each path with the entire space grid
                               False: default for training, where eval_points are the unique observed space points for each path
        :param plot: bool, whether the call is for the purpose of plotting the cond. exp. paths. for a subset of the batch
                            True: sets return_path=True, space_coords=entire grid (expected), until_T=True and changes the eval_points of the
                                  paths to be plotted from unique->entire grid, keeping unique points for all the non-plotted paths
        :param paths_to_plot: tensor of int, indices of batch paths to plot when plot = True. Defaults to only the first path.
        :param space_coords: None or (if evaluate or plot are True) list of floats, space coordinates across the space grid
        :param until_T: bool, False for training, true for evaluation and plotting. Whether to simulate conditional expectation to final time or not.

        :return: hT, loss
        '''
        # get the operator loss by default 
        LOSS = LOSS_FUN_DICT[which_loss]
        batch_size = len(eval_ptr) - 1  # nr of batches in the data batch

        # Adjust eval points if evaluating or plotting. Default is unique observed space points for each batch path (for calculating loss)
        if evaluate:
            return_path = True
            assert space_coords is not None
            until_T = True
            # overwrite the evaluation points to be the entire space grid for all batches
            space_grid = len(space_coords)
            eval_points = space_coords.repeat(batch_size)
            eval_ptr = space_grid * torch.arange(batch_size + 1)
            eval_lengths = torch.tensor(np.repeat(space_grid, batch_size))

            path_t = []
            path_y = []
        elif plot:
            assert space_coords is not None
            assert paths_to_plot is not None
            until_T = True
            space_grid = len(space_coords)
            paths_to_plot = list(paths_to_plot)
            # adjust eval points for the paths to plot, so that these paths of the batch use the entire space grid
            # last of the batches
            first_path = paths_to_plot[0]
            last_path = paths_to_plot[-1]
            # batches before the first after the last batch unaffected
            start = eval_points[:eval_ptr[first_path]]
            end = eval_points[eval_ptr[last_path + 1]:]
            # lengths of unique eval points for each batch path
            eval_lengths = torch.tensor(eval_ptr[1:] - eval_ptr[:-1])
            # eval points in need of splitting up (from first path to last path, in between, some plotted some not)
            to_split = eval_points[eval_ptr[first_path]:eval_ptr[last_path + 1]]
            lengths_split = eval_lengths[first_path:last_path + 1]
            split_points = list(torch.split(to_split, tuple(lengths_split.long())))
            # now replace split points corresponding with paths to plot with the entire grid
            split_indices = np.array(paths_to_plot) - first_path
            for idx in split_indices:
                split_points[idx] = space_coords
                lengths_split[idx] = len(space_coords)
            # recombine
            eval_points = torch.cat([start] + list(split_points) + [end], dim=0)
            # recompute eval_ptr
            all_lengths = torch.cat([eval_lengths[:first_path], lengths_split, 
                                     eval_lengths[last_path + 1:]], dim=0)
            eval_ptr = torch.cumsum(torch.cat([torch.tensor([0]), all_lengths]), dim=0)

            path_t = []
            path_y = []
        else:
            eval_lengths = torch.tensor(eval_ptr[1:] - eval_ptr[:-1])

        nr_hidden_states = len(eval_points)  # total nr of hidden states to compute across batches and space points

        if dim_to is None:
            dim_to = len(self.output_coords)
        out_coords = self.output_coords[:dim_to]

        # //  initialize //
         # takes t0=0 as initial time
        loss = torch.tensor(0.0).to(self.device)
        current_time = 0.0
        tau = torch.zeros(nr_hidden_states, 1, dtype=torch.float32).to(self.device)  # initial last observation time tau(t0)=0 for all batches
        h_init = torch.zeros(nr_hidden_states, self.hidden_size, dtype=torch.float32).to(self.device)  # all hidden states to compute
        t_0 = torch.zeros(nr_hidden_states, 1, dtype=torch.float32).to(self.device)
        c = torch.zeros(nr_hidden_states, self.context_size, dtype=torch.float32).to(self.device)

        # // Now to calculate H0: requires genk context at t0 and h0- //
        # Genk takes the processed data vectors at time t0, must therefore start by processing this data

        # -- Get the t0 data --
        # INCLUDED t_0 data in the collatefn -> i=0 to i=1 for ti. Now get t0 data:
        start = time_ptr[0]
        end = time_ptr[1]
        X_obs = X[start:end][:, out_coords] # all the path groups in space seen at t0
        space_points_0 = space_points[start:end] # space points of each observation in X_obs at t0
        # Path indices and the J_i for each batch at t0
        batch_start = batch_ptr[0] # batch_ptr here tracks nr of space points belonging to a certain batch path at time t0
        batch_end = batch_ptr[1]
        nr_batches_0 = batch_end - batch_start
        start_obs = obs_idx[batch_start:batch_end] # path indices for each path of the batch seen at t0
        n_space_0 = n_space_ot[batch_start:batch_end] # nr of space points for each obs path at t0
        n_obs_0 = n_obs_ot[start_obs]

        # -- Get the observed eval point indices at t0 for each observed path --
        # want the eval points for each path
        pw_eval_points = torch.split(eval_points, tuple(eval_lengths.long()))  
        pw_eval_points_0 = [pw_eval_points[p] for p in start_obs] 
        # and the observed points of each path
        pw_space_points_0 = torch.split(space_points_0, tuple(n_space_0.long())) 
        # get the offset for these observations (to the batch they are in)
        obs_ptr_offset = eval_ptr[start_obs]
        # getting the unique observed points at t0
        pw_obs_eval_indices = []
        for eval, obs, offset in zip(pw_eval_points_0, pw_space_points_0, obs_ptr_offset):
            # Process in chunks if eval is very large
            if len(eval) > 5000:  # threshold for chunking
                chunk_size = 2000
                indices_chunks = []
                for i in range(0, len(eval), chunk_size):
                    eval_chunk = eval[i:i+chunk_size]
                    chunk_indices = torch.argmin(
                        torch.abs(eval_chunk.unsqueeze(1) - obs.unsqueeze(0)), 
                        dim=0
                    ) + i  # offset within eval
                    indices_chunks.append(chunk_indices)
                indices = torch.cat(indices_chunks, dim=0) + offset
            else:
                indices = torch.argmin(
                    torch.abs(eval.unsqueeze(1) - obs.unsqueeze(0)), 
                    dim=0
                ) + offset
            pw_obs_eval_indices.append(indices)
        # combine to get all the indices
        obs_eval_indices = torch.cat(pw_obs_eval_indices, dim=0).cpu() # dimension [nr_obs_Xij_at_t0, ] = [nr_obs_hidden_states, ]
        obs_eval_points = eval_points[obs_eval_indices].unsqueeze(1)  # dimension [nr_obs_Xij_at_t0, 1]
        nr_obs_hidden_states = len(obs_eval_indices)  # nr of observed hidden states at t0 across all batches
        
        # -- Indices of all evaluation points to be updated at t0 -- (any batch path with an observation at t0 will have its eval points updated)
        pw_eval_indices_0 = [np.arange(eval_ptr[p], eval_ptr[p+1]) for p in start_obs]  
        eval_indices_0 = np.concatenate(pw_eval_indices_0, axis=0)  
        eval_points_0 = eval_points[eval_indices_0].unsqueeze(1)  
        # pointer for which point belongs to which path
        eval_lengths_0 = [len(pinds) for pinds in pw_eval_indices_0]  
        eval_ptr_0 = np.cumsum(np.concatenate(([0], eval_lengths_0)), axis=0) 
        nr_eval_hidden_states = len(eval_indices_0)  # nr of eval hidden states at t0 across observed batches (including repeats for non-plotted paths)

        # -- calculate h0- and ybj at t0 --
        # INITIALIZE OVER ALL POINTS
        # all zero except eval point H0-(xi) = encoder_map(0, 0, xi, 0)
        h = self.encoder_map( # vectorized (nr_hidden_states rows,) --> outputs ([nr_hidden_states, hidden_size)
            h=h_init,
            t=t_0,
            xi=eval_points.unsqueeze(1), # expand to match nr_hidden_states
            c=c, 
        )
        y = self.readout_map(h)
        
        # Get bj observed hidden states at t0-
        Y_bj = self.readout_map(h[obs_eval_indices, :])

        # -- Impute observations at t0 if masked --
        M_obs = None
        if self.masked: 
            # impute obs. as standard if masked
            if (len(self.input_coords) == len(self.output_coords)) and \
                np.all(self.input_coords == self.output_coords):
                impute = True # can impute observations
            
            # get the M_obs at t0
            M_obs = M[start:end][:, out_coords] # dimension [nr_obs_Xij_at_t0, dim_to]
            
            # mask and impute or not
            if impute:
                X_obs_impute = X_obs * M_obs + (torch.ones_like(M_obs.long()) - M_obs) * Y_bj[:, out_coords]
            else:
                X_obs_impute = X_obs * M_obs
        else: # no mask or imputing
            X_obs_impute = X_obs

        # -- Calculate genk context at t0 --
        # get new contexts of all points for paths with observations at t0 --> for updating h and y at these points
        c = self.gen_kernel(
            xi=eval_points_0, # all evaluation points at t0 of the observed paths (default: each path's unique obs points across time, plot: obs path to plot has entire grid & other obs are unique points, evaluate: entire grid for all obs paths)
            eval_ptr=eval_ptr_0, # ptr to separate eval points by batch path (for genk internal sum over j)
            X_obs_impute_i=X_obs_impute, # observations at t0 of dim [nr_obs_hidden_states=nr_obs_Xij_at_t0, dim_to]
            space_points_i=space_points_0.unsqueeze(1), # space points for observations at t0 of dim [nr_obs_Xij_at_t0, 1] 
            n_space_i=n_space_0, # initial nr of space points at t0 [nr_batches_0, ]
            M_obs=M_obs if M_obs is not None else None)   # output: context vector at t0 of dim [nr_obs_hidden_states + others of obs paths, context_size]
            
        # -- Update hidden states at t0 across all eval points of observed paths --
        temp = h.clone()
        temp[eval_indices_0, :] = self.encoder_map(
                                        h=h[eval_indices_0, :], # H0- at all eval points of observed paths of dim [nr_eval_hidden_states_0, hidden_size] 
                                        t=torch.tensor([[0.0]]).expand(nr_eval_hidden_states, 1).to(self.device), 
                                        xi=eval_points_0, # all eval points of observed paths at which to update hidden states of dim [nr_eval_hidden_states_0, 1]
                                        c=c)
        h=temp
        
        # -- readouts at observed points after jump --
        # updated readouts at observed points after jump, dim [nr_obs_hidden_states, dim_to]
        Y = self.readout_map(h[obs_eval_indices, :])  
        
        # // Calculate Y0- and Y0 for training, evaluation, or plotting //
        if evaluate:
            # readout values
            y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]

            # storing observed path after jump 
            y_store = y.view(batch_size, space_grid, -1).permute(0, 2, 1)  # [batch_size, dim, space_grid]

            # store jump values
            path_t.append(current_time)
            path_y.append(y_store)
        elif plot: 
            # get eval point indices for the points (space grid) of paths to plot
            plot_eval_indices = np.concatenate([np.arange(eval_ptr[p], eval_ptr[p+1]) for p in paths_to_plot]) # entire space grids locations

            y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]

            # update predictions for plot and store
            y_plot = y[plot_eval_indices, :]  # after jump for paths to plot, now including update
            nr_plotted_paths = len(paths_to_plot)
            y_store = y_plot.view(nr_plotted_paths, space_grid, -1).permute(0, 2, 1)  # [nr_plotted_paths, dim, space_grid]
            path_t.append(current_time)
            path_y.append(y_store)

            # calculate loss at t0
            loss += LOSS(X_obs=X_obs_impute, Y_obs=Y, Y_obs_bj=Y_bj, 
                         n_obs_ot=n_obs_0, n_space_ot=n_space_0, 
                         M_obs=M_obs if M_obs is not None else None, batch_size=batch_size, dim_to=dim_to)
        else: # training
            # only need Y_bj and Y for loss calculation
            loss += LOSS(X_obs=X_obs_impute, Y_obs=Y, Y_obs_bj=Y_bj, 
                         n_obs_ot=n_obs_0, n_space_ot=n_space_0, 
                         M_obs=M_obs if M_obs is not None else None, 
                         batch_size=batch_size, dim_to=dim_to)
        

        # // NOW iterate through time, applying the evolution, jump, and readout networks appropriately //
        additional_times = times[1:] # ignore t0 = 0, done above
        for i, obs_time in enumerate(additional_times, 1):
            print("loop number: {}, obs_time: {}, current_time: {}".format(i, obs_time, current_time))
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t - 1e-10:
                    delta_t_ = delta_t
                    bj_val = False
                else:
                    delta_t_ = obs_time - current_time
                    bj_val = True

                # evolve all hidden states via NODE
                h , current_time = self.ode_step(
                    h=h, # dim [nr_hidden_states, hidden_size]
                    current_time=current_time, # dim [1,]
                    xi=eval_points,  # dim [nr_hidden_states, space_dimension]
                    tau=tau, # dim [nr_hidden_states, 1]
                    delta_t=delta_t_
                ) # evolved hidden states and updated current time += delta_t_

                if evaluate:
                    y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]
                    if not bj_val:
                        y_store = y.view(batch_size, space_grid, -1).permute(0, 2, 1)  # [batch_size, dim, space_grid]
                        path_t.append(current_time) 
                        path_y.append(y_store)
                elif plot:
                    # store the y values for paths to plot
                    y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]
                    if not bj_val: # no need to store path of bj values(same time)
                        y_plot = y[plot_eval_indices, :]  # for paths to plot
                        y_store = y_plot.view(nr_plotted_paths, space_grid, -1).permute(0, 2, 1)  # [nr_plotted_paths, dim, space_grid]
                        path_t.append(current_time) 
                        path_y.append(y_store)
                else:
                    if bj_val:
                        # readout for bj value at obs_time
                        y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]

            # // Reached an observation: Collect all relevant data at ti //
            # -- standard data at ti --
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end][:, out_coords] 
            space_points_i = space_points[start:end] 
            batch_start = batch_ptr[i] 
            batch_end = batch_ptr[i + 1]
            nr_batches_i = batch_end - batch_start
            i_obs = obs_idx[batch_start:batch_end] 
            n_space_i = n_space_ot[batch_start:batch_end] 
            n_obs_i = n_obs_ot[i_obs]
            space_ptr = np.cumsum(np.concatenate(([0], n_space_i.cpu())), dtype='int')
            
            # -- get indices of observed eval points at ti for each observed path --
            pw_eval_points_i = [pw_eval_points[p] for p in i_obs]
            pw_space_points_i = torch.split(space_points_i, tuple(n_space_i.long())) 
            obs_ptr_offset = eval_ptr[i_obs]

            pw_eval_indices_i = [np.arange(eval_ptr[p], eval_ptr[p+1]) for p in i_obs]  
            eval_indices_i = np.concatenate(pw_eval_indices_i, axis=0)  
            eval_points_i = eval_points[eval_indices_i].unsqueeze(1)  
            eval_lengths_i = [len(pinds) for pinds in pw_eval_indices_i] 
            eval_ptr_i = np.cumsum(np.concatenate(([0], eval_lengths_i)), axis=0)  

            pw_obs_eval_indices = []
            for eval, obs, offset in zip(pw_eval_points_i, pw_space_points_i, obs_ptr_offset):
                # Process in chunks if eval is very large
                if len(eval) > 5000:  # threshold for chunking
                    chunk_size = 2000
                    indices_chunks = []
                    for i in range(0, len(eval), chunk_size):
                        eval_chunk = eval[i:i+chunk_size]
                        chunk_indices = torch.argmin(
                            torch.abs(eval_chunk.unsqueeze(1) - obs.unsqueeze(0)), 
                            dim=0
                        ) + i  # offset within eval
                        indices_chunks.append(chunk_indices)
                    indices = torch.cat(indices_chunks, dim=0) + offset
                else:
                    indices = torch.argmin(
                        torch.abs(eval.unsqueeze(1) - obs.unsqueeze(0)), 
                        dim=0
                    ) + offset
                pw_obs_eval_indices.append(indices)

            # combine to get all the indices
            obs_eval_indices = torch.cat(pw_obs_eval_indices, dim=0).cpu() 
            obs_eval_points = eval_points[obs_eval_indices].unsqueeze(1)  
            nr_obs_hidden_states = len(obs_eval_indices)  
            nr_eval_hidden_states = len(eval_indices_i)

            # // Calculate Y_bj and H update at observed points //
            # -- bj observed hidden states and readouts at ti- --
            Y_bj = self.readout_map(h[obs_eval_indices, :]) 

            # -- Impute observations at ti if masked --
            if self.masked:
                M_obs = M[start:end][:, out_coords] 
                # mask and impute or not
                if impute:
                    X_obs_impute = X_obs * M_obs + (torch.ones_like(M_obs.long()) - M_obs) * Y_bj[:, out_coords]
                else:
                    X_obs_impute = X_obs * M_obs
            else: # no mask or imputing
                X_obs_impute = X_obs

            # -- Calculate genk context at ti --
            c = self.gen_kernel(
                xi=eval_points_i,
                eval_ptr=eval_ptr_i, 
                X_obs_impute_i=X_obs_impute,
                space_points_i=space_points_i.unsqueeze(1),
                n_space_i=n_space_i,
                M_obs=M_obs if self.masked else None)
            # -- update hidden state at ti across all eval points of observed paths --
            temp = h.clone()
            temp[eval_indices_i, :] = self.encoder_map(
                                        h=h[eval_indices_i, :], # Hti- at all eval points of observed paths of dim [nr_eval_hidden_states_i, hidden_size] 
                                        t=torch.tensor([[obs_time]], dtype=torch.float32).expand(nr_eval_hidden_states, 1).to(self.device), 
                                        xi=eval_points_i, # all eval points of observed paths at which to update hidden states of dim [nr_eval_hidden_states_i, 1]
                                        c=c)
            h=temp

            # -- readouts at observed points after jump --
            Y = self.readout_map(h[obs_eval_indices, :])  # updated readouts at observed points after jump, dim [nr_obs_hidden_states, dim_to]
           
            # -- loss, evaluate, or plot calculations --
            if evaluate: # evaluate model relative to true cond. exp, no loss
                path_t.append(current_time)
                y = self.readout_map(h)  
                
                # reshape to [batch_size, dim, space_grid] for storing bj paths
                y_store = y.view(batch_size, space_grid, -1).permute(0, 2, 1)  # [batch_size, dim, space_grid]
                path_y.append(y_store)
            elif plot: # plot paths and calc loss
                path_t.append(current_time)
                y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]
                
                # reshape to [nr_plotted_paths, dim, space_grid] for storing paths to plot
                y_plot = y[plot_eval_indices, :]  # for paths to plot
                y_store = y_plot.view(nr_plotted_paths, space_grid, -1).permute(0, 2, 1)  # [nr_plotted_paths, dim, space_grid]
                path_y.append(y_store)

                loss += LOSS(X_obs=X_obs_impute, Y_obs=Y, Y_obs_bj=Y_bj,
                                n_obs_ot=n_obs_i, n_space_ot=n_space_i,
                                M_obs=M_obs if M_obs is not None else None, batch_size=batch_size, dim_to=dim_to)
            else: # only loss
                loss += LOSS(X_obs=X_obs_impute, Y_obs=Y, Y_obs_bj=Y_bj,
                                n_obs_ot=n_obs_i, n_space_ot=n_space_i,
                                M_obs=M_obs if M_obs is not None else None, batch_size=batch_size, dim_to=dim_to)
            
            # update tau for observed hidden states
            tau_i = torch.tensor([[obs_time]], dtype=torch.float32, device=tau.device)
            tau = tau.clone()
            tau[eval_indices_i, :] = tau_i


        print("reached final observation time")
        # after every observation has been processed, propagating until T
        if until_T:
            # Calculate conditional expectation stepwise
            while current_time < (T - 1e-10*delta_t):
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time

                # evolve all hidden states via NODE
                h , current_time = self.ode_step(
                    h=h,
                    current_time=torch.tensor([[current_time]]).expand(nr_hidden_states, 1).to(self.device), 
                    xi=eval_points, 
                    tau=tau, 
                    delta_t=delta_t_
                ) # evolved hidden states and updated current time += delta_t_

                if evaluate:
                    path_t.append(current_time)
                    y = self.readout_map(h)  # after jump for all paths, dimension [nr_hidden_states, dim_to]
                    y_store = y.view(batch_size, space_grid, -1).permute(0, 2, 1)  # [batch_size, dim, space_grid]
                    path_y.append(y_store)
                elif plot:
                    # store the y values for paths to plot
                    path_t.append(current_time)
                    y = self.readout_map(h)
                    y_plot = y[plot_eval_indices, :]  # for paths to plot
                    y_store = y_plot.view(nr_plotted_paths, space_grid, -1).permute(0, 2, 1)  # [nr_plotted_paths, dim, space_grid]
                    path_y.append(y_store)
            print("reached final time T")

        path_s = None # not implemented
        path_h = None # not implemented

        if evaluate: # does not calculate loss
            return h, np.array(path_t), path_s, path_h, torch.stack(path_y)[:, :, :dim_to]  # path_y is [time_steps, batch_size, dim, space_grid]
        elif plot:
            return h, loss, np.array(path_t), path_s, path_h, torch.stack(path_y)[:, :, :dim_to] # path_y is [time_steps, nr_plotted_paths, dim, space_grid]
        else:
            return h, loss
        
    def evaluate(self, eval_points, eval_ptr, times, time_ptr, X, obs_idx, 
                 space_points, n_space_ot, batch_ptr, time_idx, space_idx, observed_dates,
                 start_X, start_M,
                 delta_t, T, n_obs_ot, stockmodel, space_coords=None,
                 diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
                 M=None, true_paths=None, true_mask=None, mult=None,
                 use_stored_cond_exp=False, return_paths=False):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        """
        self.eval() # eval mode

        dim = X.shape[1] # start_X is [batch_size, dim, space_grid]
        dim_to = dim # all dimensions
        output_dim_to = len(self.output_coords) # includes all the output coords as opposed to out_coords in forward which only takes the first dim_to (input in forward) coords of the output coords

        _, path_t, path_s, path_h, path_y = self.forward(
            eval_points, eval_ptr, times, time_ptr, X, obs_idx, 
            space_points, n_space_ot, batch_ptr, 
            delta_t, T, n_obs_ot=n_obs_ot,
            evaluate=True, space_coords=space_coords, until_T=True, M=M,
            dim_to=output_dim_to)

        # involve calculation over the appropriate space coords not just at times
        if true_paths is None: # use conditional expectation for comparison
            if M is not None:
                M = M.detach().cpu().numpy()[:, :dim_to] # ? if dim_to=dim, the data dimension, then these are the same.
                start_M = start_M.detach().cpu().numpy()
            if X.shape[0] > 0:  # if no data (eg. bc. obs_perc=0, not possible)
                X = X.detach().cpu().numpy()[:, :dim_to]
            _, true_path_t, true_path_s, true_path_y = stockmodel.compute_cond_exp(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X.detach().cpu().numpy()[:, :dim_to], 
                    n_obs_ot.detach().cpu().numpy(), space_points=space_points.detach().cpu().numpy(), observed_dates=observed_dates, 
                    n_space_ot=n_space_ot.detach().cpu().numpy(), batch_ptr=batch_ptr, time_idx=time_idx, space_idx=space_idx, 
                    start_M=start_M, return_path=True, get_loss=False, M=M, store_and_use_stored=use_stored_cond_exp) 
            # here outputs match path_y shape and values (all on time x space observations) [time_steps, batch_size, dim_to, space_grid]
        else: # use true paths
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t))) # finds the index of the true_t closest to each t in path_t. It is of dimension
            true_path_t = true_t[which_t_ind]
            true_path_y = true_paths[:, self.output_coords[:output_dim_to], :, :][:, :, which_t_ind, :] # [batch_size, dim_to, time_steps, space_grid]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1, 3))

        if path_y.detach().cpu().numpy().shape == true_path_y.shape:
            eval_metric = diff_fun(path_y.detach().cpu().numpy(), true_path_y) # just need to change the diff_fun here --> now path_y & true_path_y are [time_steps, space_grid, batch_size, dim_to]
        else:
            print(path_y.detach().cpu().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_metric, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_metric
        
    def get_pred(self, eval_points, eval_ptr,times, time_ptr, X, obs_idx, delta_t, T, n_obs_ot, 
                space_points, n_space_ot, batch_ptr, space_coords=None, paths_to_plot=None,
                 M=None, which_loss=None):
        """
        get predicted path: ONJODE forward pass with plotting enabled
        """
        self.eval()
        h, loss, path_t, path_s, path_h, path_y = self.forward(
            eval_points=eval_points, eval_ptr=eval_ptr, 
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            space_points=space_points, n_space_ot=n_space_ot, batch_ptr=batch_ptr, 
            delta_t=delta_t, T=T, n_obs_ot=n_obs_ot,
            plot=True, evaluate=False, paths_to_plot=paths_to_plot, space_coords=space_coords, until_T=True, M=M,
            which_loss=which_loss)

        return {'pred': path_y, 'pred_t': path_t, 'pred_s': path_s, 'pred_h': path_h, 'loss': loss}
        

class NJODE(torch.nn.Module):
    """
    NJ-ODE model
    """
    def __init__(  # initialize the class by naming relevant features
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            input_coords=None, output_coords=None,
            signature_coords=None, compute_variance=None, var_size=0,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param input_coords: list of int, the coordinates of the input
        :param output_coords: list of int, the coordinates of the output
        :param signature_coords: list of int, the coordinates of the signature
        :param compute_variance: None or one of {"variance", "covariance"},
                whether to compute the (marginal) variance or covariance matrix
        :param var_size: int, the size of the model variance estimate; this is
                already included in the output_size, but the variance
                coordinates are not included in the output_coords
        :param options: kwargs, used:
                - "classifier_nn"
                - "options" with arg a dict passed from train.train
                    used kwords: 'which_loss', 'residual_enc_dec',
                    'residual_enc', 'residual_dec',
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode', 'enc_input_t', 'use_current_y_for_ode',
                    'use_observation_as_input', 'coord_wise_tau', 'clamp',
                    'ode_input_scaling_func', 'use_sig_for_classifier',
                    'classifier_loss_weight'
        """
        super().__init__()  # super refers to base class, init initializes

        self.epoch = 1
        self.retrain_epoch = 0
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn  # use RNN for jumps
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.signature_coords = signature_coords
        self.compute_variance = compute_variance
        self.var_size = var_size
        self.var_weight = 1.
        self.which_var_loss = None


        # get options from the options of train input
        options1 = options['options']
        if 'which_loss' in options1:
            self.which_loss = options1['which_loss']
        else:
            self.which_loss = 'standard'  # otherwise take the standard loss
        assert self.which_loss in LOSS_FUN_DICT
        print('using loss: {}'.format(self.which_loss))
        self.loss_quantiles = None
        self.nb_quantiles = None
        if "quantile" in self.which_loss:
            self.loss_quantiles = options1['loss_quantiles']
            self.nb_quantiles = len(self.loss_quantiles)
            print("using quantile loss with quantiles:", self.loss_quantiles)
        if "var_weight" in options1:
            self.var_weight = options1['var_weight']
            print("using variance loss weight:", self.var_weight)
        if "which_var_loss" in options1:
            self.which_var_loss = options1['which_var_loss']
            print("using variance loss:", self.which_var_loss)

        self.residual_enc = True
        self.residual_dec = True
        # for backward compatibility, set residual_enc to False as default
        #   if RNN is used. (before, it was not possible to use residual
        #   connections with RNNs)
        if self.use_rnn:
            self.residual_enc = False
        if 'residual_enc_dec' in options1:
            residual_enc_dec = options1['residual_enc_dec']
            self.residual_enc = residual_enc_dec
            self.residual_dec = residual_enc_dec
        if 'residual_enc' in options1:
            self.residual_enc = options1['residual_enc']
        if 'residual_dec' in options1:
            self.residual_dec = options1['residual_dec']

        self.input_current_t = False
        if 'input_current_t' in options1:
            self.input_current_t = options1['input_current_t']
        self.input_var_t_helper = False
        if 'input_var_t_helper' in options1:
            self.input_var_t_helper = options1['input_var_t_helper']
        self.input_sig = False
        if 'input_sig' in options1:
            self.input_sig = options1['input_sig']
        self.level = 2
        if 'level' in options1:
            self.level = options1['level']
        self.sig_depth = sig.siglength(len(self.signature_coords)+1, self.level)
        self.masked = False
        if 'masked' in options1:
            self.masked = options1['masked']
        self.use_y_for_ode = False
        if 'use_y_for_ode' in options1:
            self.use_y_for_ode = options1['use_y_for_ode']
        self.use_current_y_for_ode = False
        if 'use_current_y_for_ode' in options1:
            self.use_current_y_for_ode = options1['use_current_y_for_ode']
        if self.nb_quantiles is not None and self.nb_quantiles > 1:
            assert self.use_current_y_for_ode is False, \
                "Quantile loss not implemented for use_current_y_for_ode"
            assert self.use_y_for_ode is False, \
                "Quantile loss not implemented for use_y_for_ode"
        self.coord_wise_tau = False
        if 'coord_wise_tau' in options1 and self.masked:
            self.coord_wise_tau = options1['coord_wise_tau']
        self.enc_input_t = False
        if 'enc_input_t' in options1:
            self.enc_input_t = options1['enc_input_t']
        self.clamp = None
        if 'clamp' in options1:
            self.clamp = options1['clamp']
        self.ode_input_scaling_func = "tanh"
        if 'ode_input_scaling_func' in options1:
            self.ode_input_scaling_func = options1['ode_input_scaling_func']
        classifier_dict = None
        if 'classifier_dict' in options:
            classifier_dict = options["classifier_dict"]
        self.use_sig_for_classifier = False
        if 'use_sig_for_classifier' in options1:
            self.use_sig_for_classifier = options1['use_sig_for_classifier']
        self.class_loss_weight = 1.
        self.loss_weight = 1.
        if 'classifier_loss_weight' in options1:
            class_loss_weight = options1['classifier_loss_weight']
            if class_loss_weight == np.infty:
                self.class_loss_weight = 1.
                self.loss_weight = 0.
            else:
                self.class_loss_weight = class_loss_weight
        t_size = 2
        if self.coord_wise_tau:
            t_size = 2*input_size
        use_observation_as_input = None
        if 'use_observation_as_input' in options1:
            use_observation_as_input = options1['use_observation_as_input']
        if use_observation_as_input is None:
            self.use_observation_as_input = lambda x: True
        elif isinstance(use_observation_as_input, bool):
            self.use_observation_as_input = \
                lambda x: use_observation_as_input
        elif isinstance(use_observation_as_input, float):
            self.use_observation_as_input = \
                lambda x: np.random.random() < use_observation_as_input
        elif isinstance(use_observation_as_input, str):
            self.use_observation_as_input = \
                eval(use_observation_as_input)
        val_use_observation_as_input = None
        if 'val_use_observation_as_input' in options1:
            val_use_observation_as_input = \
                options1['val_use_observation_as_input']
        if val_use_observation_as_input is None:
            self.val_use_observation_as_input = self.use_observation_as_input
        elif isinstance(val_use_observation_as_input, bool):
            self.val_use_observation_as_input = \
                lambda x: val_use_observation_as_input
        elif isinstance(val_use_observation_as_input, float):
            self.val_use_observation_as_input = \
                lambda x: np.random.random() < val_use_observation_as_input
        elif isinstance(val_use_observation_as_input, str):
            self.val_use_observation_as_input = \
                eval(val_use_observation_as_input)

        self.ode_f = ODEFunc(
            input_size=input_size, hidden_size=hidden_size, ode_nn=ode_nn,
            dropout_rate=dropout_rate, bias=bias,
            input_current_t=self.input_current_t, input_sig=self.input_sig,
            sig_depth=self.sig_depth, coord_wise_tau=self.coord_wise_tau,
            input_scaling_func=self.ode_input_scaling_func,
            use_current_y_for_ode=self.use_current_y_for_ode,
            input_var_t_helper=self.input_var_t_helper)
        self.encoder_map = FFNN(
            input_size=input_size, output_size=hidden_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc,
            input_sig=self.input_sig, sig_depth=self.sig_depth,
            input_t=self.enc_input_t, t_size=t_size) # jump network
        self.readout_map = FFNN(
            input_size=hidden_size, output_size=output_size, nn_desc=readout_nn,
            dropout_rate=dropout_rate, bias=bias,
            residual=self.residual_dec, clamp=self.clamp,
            nb_outputs=self.nb_quantiles)
        self.get_classifier(classifier_dict=classifier_dict)

        self.solver = solver
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def get_classifier(self, classifier_dict):
        self.classifier = None
        self.SM = None
        self.CEL = None
        if classifier_dict is not None:
            if self.use_sig_for_classifier:
                classifier_dict['input_size'] += self.sig_depth
            self.classifier = FFNN(**classifier_dict)
            self.SM = torch.nn.Softmax(dim=1)
            self.CEL = torch.nn.CrossEntropyLoss()

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None,
                 current_y=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        if self.solver == "euler":
            h = h + delta_t * self.ode_f(
                x=last_X, h=h, tau=tau, tdiff=current_time - tau,
                signature=signature, current_y=current_y, delta_t=delta_t)
        else:
            raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def recreate_data(self, times, time_ptr, X, obs_idx, start_X):
        """
        recreates matrix of all observations
        first dim: which data path
        second dim: which time
        """
        # shape: [batch_size, time_steps+1, dimension]
        data = np.empty(shape=(start_X.shape[0], 1+len(times), start_X.shape[1]))
        data[:] = np.nan
        data[:,0,:] = start_X.detach().cpu().numpy()

        X = X.detach().cpu().numpy()
        for j, time in enumerate(times):
            start = time_ptr[j]
            end = time_ptr[j + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            data[i_obs, j+1, :] = X_obs
        times_new = np.concatenate(([0], times), axis=0)

        return times_new, data

    def get_signature(self, times, time_ptr, X, obs_idx, start_X):
        """
        Input: See forward
        Returns: signature of paths as nested list
        """
        # reconstructing the data, shape: [batch_size, time_steps+1, dim]
        times_new, data = self.recreate_data(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            start_X=start_X)

        # list of list of lists, shape: [batch_size, obs_dates[j], sig_length]
        signature = []
        for j in range(data.shape[0]):  # iterate over batch
            data_j = data[j, :, :]
            observed_j = []
            for i in range(data_j.shape[0]):
                # if the current batch-sample has an observation at the current
                #   time, add it to the list of observations
                if not np.all(np.isnan(data_j[i])):
                    observed_j += [i]
            data_j = data_j[observed_j, :]

            # replace no observations with last observation
            for i in range(1, data_j.shape[0]):
                # # OLD VERSION (SLOW)
                # for k in range(data_j.shape[1]):
                #     if np.isnan(data_j[i, k]):
                #         data_j[i, k] = data_j[i-1, k]
                ks = np.isnan(data_j[i, :])
                data_j[i, ks] = data_j[i-1, ks]

            times_j = times_new[observed_j].reshape(-1, 1)
            # add times to data for signature call
            path_j = np.concatenate((times_j, data_j), axis=1)
            # the following computes the signatures of all partial paths, from
            #   start to each point of the path
            signature.append(sig.sig(path_j, self.level, 2))

        return signature

    def forward(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=False, get_loss=True, until_T=False,
                M=None, start_M=None, which_loss=None, dim_to=None,
                predict_labels=None, return_classifier_out=False,
                return_at_last_obs=False, compute_variance_loss=True):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :param return_at_last_obs: bool, whether to return the hidden state at
                the last observation time or at the final time
        :param epoch: int, the current epoch
        :param compute_variance_loss: bool, whether to compute the variance
                loss (in case self.compute_variance is not None). default: True

        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """
        if which_loss is None:
            which_loss = self.which_loss
        if 'quantile' in which_loss:
            LOSS = LOSS_FUN_DICT[which_loss](self.loss_quantiles)
        else:
            LOSS = LOSS_FUN_DICT[which_loss]

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = len(self.output_coords)
        out_coords = self.output_coords[:dim_to]

        impute = False
        if (len(self.input_coords) == len(self.output_coords) and
                np.all(self.input_coords == self.output_coords) and
                self.loss_quantiles is None):
            impute = True
        if not impute and self.use_y_for_ode:
            raise ValueError(
                "use_y_for_ode can only be used when imputation is possible, "
                "i.e., when input and output coordinates are the same")

        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, self.input_size).to(
                self.device)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1).to(
                self.device)
        current_time = 0.0
        loss = torch.tensor(0.).to(self.device)
        c_sig = None

        if (self.input_sig or self.use_sig_for_classifier):
            if X.shape[0] == 0:  # if no data, set signature to 0
                pass
            elif self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr,
                    X=X_obs_impute[:, self.signature_coords],
                    obs_idx=obs_idx, start_X=start_X[:, self.signature_coords])
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr,
                    X=X[:, self.signature_coords],
                    obs_idx=obs_idx, start_X=start_X[:, self.signature_coords])

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float().to(self.device)

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
                start_M = start_M[:, self.input_coords]
        else:
            start_M = None

        # initial hidden state (t0)
        h = self.encoder_map( # batch_size x dim --> batch_size x hidden_size
            start_X[:, self.input_coords], mask=start_M, # evaluated at input coords across batches
            sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)).to(self.device),
            t=torch.cat((tau, current_time - tau), dim=1).to(self.device))
        # if self.encoder_map.use_lstm:
        #     self.c_ = torch.chunk(h.clone(), chunks=2, dim=1)[1]

        if return_path:
            path_t = [0]
            path_h = [h]
            y = self.readout_map(h)
            if self.var_size > 0:
                path_y = [y[:, :-self.var_size]]
                path_var = [y[:, -self.var_size:]]
            else:
                path_y = [y]
                path_var = None
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time,
                        last_X=last_X[:, self.input_coords], # evaluated at input coords across batches
                        tau=tau,
                        signature=c_sig, current_y=self.readout_map(h))
                    current_time_nb = int(round(current_time / delta_t)) # outputs h over batches at next time step
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    y = self.readout_map(h)
                    if self.var_size > 0:
                        path_y.append(y[:, :-self.var_size])
                        path_var.append(y[:, -self.var_size:])
                    else:
                        path_y.append(y)

            # Reached an observation - only update those elements of the batch, 
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end] 
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end]).to(self.device)
                else:
                    M_obs = M[start:end]
                M_obs_in = M_obs[:, self.input_coords]
                M_obs_out = M_obs[:, out_coords]
                M_obs_sig = M_obs[:, self.signature_coords]
            else:
                M_obs = None
                M_obs_in = None
                M_obs_out = None
                M_obs_sig = None

            # decide whether to use observation as input
            if self.training:  # check whether model is in training or eval mode
                use_as_input = self.use_observation_as_input(self.epoch)
            else:
                use_as_input = self.val_use_observation_as_input(self.epoch)

            # update signature
            if self.input_sig or self.use_sig_for_classifier:
                for ij, j in enumerate(i_obs):
                    # the signature is updated only if one of the sig-coords is
                    #   observed -> hence, it can happen that even though the
                    #   j-th batch sample is observed, the signature is not
                    #   updated because none of the sig-coords is observed
                    if M_obs_sig is None or M_obs_sig[ij].sum() > 0:
                        current_sig[j, :] = signature[j][current_sig_nb[j]]
                        current_sig_nb[j] += 1
                if use_as_input:
                    # TODO: this is not fully correct, since if we didn't
                    #   use some intermediate observations, the signature still
                    #   has their information when using the signature up to
                    #   some later observation. However, this just means that
                    #   during training, the model conditions on a (slightly)
                    #   different sigma-algebra (if the signature is used), but
                    #   for inference the model should still work correctly.
                    #   Especially, if we are interested in predicting
                    #   \hat{X}_{t,s}
                    c_sig = torch.from_numpy(current_sig).float().to(
                        self.device)

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.readout_map(h) # from [batch_size, hidden_size] to [batch_size, output_size]
            if use_as_input:
                X_obs_impute = X_obs
                temp = h.clone()
                if self.masked:
                    if impute:
                        # self imputation only possible if input and output are
                        #    the same and no quantile loss is used
                        X_obs_impute = X_obs * M_obs + (torch.ones_like(
                            M_obs.long()) - M_obs) * Y_bj[i_obs.long(), :data_dim]
                    else:
                        # otherwise set all masked entries to last value of X
                        X_obs_impute = X_obs * M_obs + (1-M_obs) * last_X
                c_sig_iobs = None
                if self.input_sig:
                    c_sig_iobs = c_sig[i_obs]
                temp[i_obs.long()] = self.encoder_map(
                    X_obs_impute[:, self.input_coords], # dimension is [observed_data_from_batches_at_current_time, dim]
                    mask=M_obs_in, sig=c_sig_iobs, h=h[i_obs],
                    t=torch.cat((tau[i_obs], current_time - tau[i_obs]), dim=1)) # previous obs time and diff to current time (obs)
                h = temp # update hidden state over the observed batches only
                Y = self.readout_map(h) # get output at this time

                # update h and sig at last observation
                h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
                sig_at_last_obs = c_sig
            else:
                Y = Y_bj

            if get_loss:
                Y_var_bj = None
                Y_var = None
                if self.var_size > 0:
                    Y_var_bj = Y_bj[i_obs.long(), -self.var_size:]
                    Y_var = Y[i_obs.long(), -self.var_size:]

                # INFO: X_obs has input and output coordinates, out_coords only
                #   has the output coordinates until dim_to; Y_obs has only the
                #   output coordinates (+ the var coords appended in the end),
                #   so taking them until dim_to (which is at max the size of the
                #   output_coords) corresponds to the out_coords
                if compute_variance_loss:
                    compute_variance = self.compute_variance
                else:
                    compute_variance = None
                loss = loss + LOSS(
                    X_obs=X_obs[:, out_coords], Y_obs=Y[i_obs.long(), :dim_to],
                    Y_obs_bj=Y_bj[i_obs.long(), :dim_to],
                    n_obs_ot=n_obs_ot[i_obs.long()], batch_size=batch_size,
                    weight=self.weight, M_obs=M_obs_out,
                    compute_variance=compute_variance,
                    var_weight=self.var_weight,
                    Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to,
                    which_var_loss=self.which_var_loss)

            # make update of last_X and tau, that is not inplace 
            # (otherwise problems in autograd)
            if use_as_input:
                temp_X = last_X.clone()
                temp_tau = tau.clone()
                if self.use_y_for_ode:
                    temp_X[i_obs.long()] = Y[i_obs.long(), :data_dim]
                else:
                    temp_X[i_obs.long()] = X_obs_impute
                if self.coord_wise_tau:
                    _M = torch.zeros_like(temp_tau)
                    _M[i_obs] = M_obs[:, self.input_coords]
                    temp_tau[_M==1] = obs_time.astype(np.float64)
                else:
                    temp_tau[i_obs.long()] = obs_time.astype(np.float64)
                last_X = temp_X
                tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                if self.var_size > 0:
                    path_y.append(Y[:, :-self.var_size])
                    path_var.append(Y[:, -self.var_size:])
                else:
                    path_y.append(Y)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=cl_out, target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig, current_y=self.readout_map(h))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    y = self.readout_map(h)
                    if self.var_size > 0:
                        path_y.append(y[:, :-self.var_size])
                        path_var.append(y[:, -self.var_size:])
                    else:
                        path_y.append(y)

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            var_path = None
            if self.var_size > 0:
                var_path = torch.stack(path_var)
            if return_classifier_out:
                return h, loss, np.array(path_t), torch.stack(path_h), \
                       torch.stack(path_y)[:, :, :dim_to], var_path, cl_out
            return h, loss, np.array(path_t), torch.stack(path_h), \
                   torch.stack(path_y)[:, :, :dim_to], var_path
        else:
            if return_classifier_out and self.classifier is not None:
                return h, loss, cl_out
            return h, loss

    def evaluate(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, stockmodel, cond_exp_fun_kwargs=None,
                 diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
                 return_paths=False, M=None, true_paths=None, start_M=None,
                 true_mask=None, mult=None, use_stored_cond_exp=False,):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :param start_M: see forward
        :param true_paths: np.array, shape [batch_size, dimension, time_steps+1]
        :param true_mask: as true_paths, with mask entries
        :param mult: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim/mult. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param use_stored_cond_exp: bool, whether to recompute the cond. exp.

        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        dim = start_X.shape[1]
        dim_to = dim
        output_dim_to = len(self.output_coords)
        if mult is not None and mult > 1:
            dim_to = round(dim/mult)
            output_dim_to = round(len(self.output_coords)/mult)

        _, _, path_t, path_h, path_y, path_var = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, None,
            return_path=True, get_loss=False, until_T=True, M=M,
            start_M=start_M, dim_to=output_dim_to)

        if true_paths is None:
            if M is not None:
                M = M.detach().cpu().numpy()[:, :dim_to]
            if X.shape[0] > 0:  # if no data (eg. bc. obs_perc=0, not possible)
                X = X.detach().cpu().numpy()[:, :dim_to]
            _, true_path_t, true_path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X,
                obs_idx.detach().cpu().numpy(),
                delta_t, T, start_X.detach().cpu().numpy()[:, :dim_to],
                n_obs_ot.detach().cpu().numpy(),
                return_path=True, get_loss=False, M=M,
                store_and_use_stored=use_stored_cond_exp)
        else:
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t)))
            # INFO: first get the correct output coordinate, then the correct
            #   time index; afterwards transpose to [time, batch_size, dim]
            true_path_y = true_paths[:, self.output_coords[:output_dim_to], :][
                :, :, which_t_ind]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1))
            true_path_t = true_t[which_t_ind]

        if path_y.detach().cpu().numpy().shape == true_path_y.shape:
            eval_metric = diff_fun(path_y.detach().cpu().numpy(), true_path_y)
        else:
            print(path_y.detach().cpu().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_metric, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_metric

    def evaluate_LOB(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_paths=False, predict_times=None,
            true_predict_vals=None, true_predict_labels=None, true_samples=None,
            normalizing_mean=0., normalizing_std=1., eval_predict_steps=None,
            thresholds=None, predict_labels=None,
            coord_to_compare=(0,), class_report=False):
        """
        evaluate the model at its current training state for the LOB dataset

        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param return_paths: bool, whether to return also the paths
        :param predict_times: np.array with the times at which each sample
                should be predicted
        :param true_predict_vals: np.array with the true values that should be
                predicted at the predict_times
        :param true_predict_labels: np.array, the correct labels at
                predict_times
        :param true_samples: np.array, the true samples, needed to compute
                predicted labels
        :param normalizing_mean: float, the mean with which the price data was
                normalized
        :param normalizing_std: float, the std with which the price data was
                normalized
        :param eval_predict_steps: list of int, the amount of steps ahead at
                which to predict
        :param thresholds: list of float, the labelling thresholds for each
                entry of eval_predict_steps
        :param predict_labels: as true_predict_labels, but as torch.tensor with
                classes in {0,1,2}
        :param coord_to_compare: list or None, the coordinates on which the
                output and input are compared, applied to the inner dimension of
                the time series, e.g. use [0] to compare on the midprice only
        :param class_report: bool, whether to print the classification report
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        bs = start_X.shape[0]
        dim = start_X.shape[1]
        if coord_to_compare is None:
            coord_to_compare = np.arange(dim)

        _, _, path_t, path_h, path_y, path_var, cl_out = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=False, until_T=True, M=None,
            start_M=None, dim_to=None, predict_labels=predict_labels,
            return_classifier_out=True)

        path_y = path_y.detach().cpu().numpy()
        predicted_vals = np.zeros_like(true_predict_vals)
        for i in range(bs):
            t = predict_times[i][0]
            t_ind = np.argmin(np.abs(path_t - t))
            predicted_vals[i, :, 0] = path_y[t_ind, i, :]

        eval_metric = np.nanmean(
            (predicted_vals[:, coord_to_compare, 0] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        ref_eval_metric = np.nanmean(
            (true_samples[:, coord_to_compare, -1] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        f1_scores = None
        predicted_labels = None
        if true_samples is not None and true_predict_labels is not None:
            predicted_labels = np.zeros(bs)
            if cl_out is not None:
                class_probs = self.SM(cl_out).detach().cpu().numpy()
                classes = np.argmax(class_probs, axis=1) - 1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], classes,
                    average="weighted")
                predicted_labels = classes
            else:
                # TODO: this computes the labels incorrectly, since the shift by
                #  X_0 is missing -> results should not be trusted, better to
                #  use classifier
                m_minus = np.mean(
                    true_samples[:, 0, -eval_predict_steps[0]:] *
                    normalizing_std + normalizing_mean, axis=1)
                m_plus = predicted_vals[:, 0, 0]*normalizing_std + \
                         normalizing_mean
                pctc = (m_plus - m_minus) / m_minus
                predicted_labels[pctc > thresholds[0]] = 1
                predicted_labels[pctc < -thresholds[0]] = -1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], predicted_labels,
                    average="weighted")
            if class_report:
                print("eval-mse: {:.5f}".format(eval_metric))
                print("f1-score: {:.5f}".format(f1_scores))
                print("classification report \n",
                      sklearn.metrics.classification_report(
                          true_predict_labels[:, 0], predicted_labels,))

        if return_paths:
            return eval_metric, ref_eval_metric, f1_scores, path_t, path_y, \
                   predicted_vals[:, :, 0], predicted_labels
        else:
            return eval_metric, f1_scores

    def get_pred(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, M=None, start_M=None, which_loss=None):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :param start_M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        h, loss, path_t, path_h, path_y, path_var = self.forward(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
            return_path=True, get_loss=True, until_T=True, M=M,
            start_M=start_M, which_loss=which_loss)
        return {'pred': path_y, 'pred_t': path_t, 'loss': loss,
                'pred_var': path_var}

    def forward_classifier(self, x, y):
        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        cl_loss = None
        if self.classifier is not None:
            cl_out = self.classifier(x)
            cl_loss = self.CEL(input=cl_out, target=y)
        return cl_loss, cl_out



class randomizedNJODE(torch.nn.Module):
    """
    NJ-ODE model
    """
    def __init__(
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param options: kwargs, used:
                - "options" with arg a dict passed
                    from train.train (kwords: 'which_loss', 'residual_enc_dec',
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode' are used)
        """
        super().__init__()  # super refers to base class, init initializes

        self.epoch = 1
        self.retrain_epoch = 0
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn  # use RNN for jumps

        # get options from the options of train input
        options1 = options['options']
        if 'which_loss' in options1:
            self.which_loss = options1['which_loss']  # change loss if specified in options
        else:
            self.which_loss = 'standard'  # otherwise take the standard loss
        assert self.which_loss in LOSS_FUN_DICT
        print('using loss: {}'.format(self.which_loss))

        self.residual_enc_dec = True
        if self.use_rnn:
            self.residual_enc_dec = False
        if 'residual_enc_dec' in options1:
            self.residual_enc_dec = options1['residual_enc_dec']
        if 'residual_enc' in options1:
            self.residual_enc_dec = options1['residual_enc']
        self.input_current_t = False
        if 'input_current_t' in options1:
            self.input_current_t = options1['input_current_t']
        self.input_sig = False
        if 'input_sig' in options1:
            self.input_sig = options1['input_sig']
        self.level = 2
        if 'level' in options1:
            self.level = options1['level']
        self.sig_depth = sig.siglength(input_size+1, self.level)
        self.masked = False
        if 'masked' in options1:
            self.masked = options1['masked']
        self.use_y_for_ode = True
        if 'use_y_for_ode' in options1:
            self.use_y_for_ode = options1['use_y_for_ode']
        self.coord_wise_tau = False
        if 'coord_wise_tau' in options1 and self.masked:
            self.coord_wise_tau = options1['coord_wise_tau']
        classifier_dict = None
        if 'classifier_dict' in options:
            classifier_dict = options["classifier_dict"]
        self.use_sig_for_classifier = False
        if 'use_sig_for_classifier' in options1:
            self.use_sig_for_classifier = options1['use_sig_for_classifier']
        self.class_loss_weight = 1.
        self.loss_weight = 1.
        if 'classifier_loss_weight' in options1:
            class_loss_weight = options1['classifier_loss_weight']
            if class_loss_weight == np.infty:
                self.class_loss_weight = 1.
                self.loss_weight = 0.
            else:
                self.class_loss_weight = class_loss_weight

        self.ode_f = ODEFunc(
            input_size=input_size, hidden_size=hidden_size, ode_nn=ode_nn,
            dropout_rate=dropout_rate, bias=bias,
            input_current_t=self.input_current_t, input_sig=self.input_sig,
            sig_depth=self.sig_depth, coord_wise_tau=self.coord_wise_tau)
        self.encoder_map = FFNN(
            input_size=input_size, output_size=hidden_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc_dec,
            input_sig=self.input_sig, sig_depth=self.sig_depth)
        self.readout_map = LinReg(hidden_size, output_size)
        self.get_classifier(classifier_dict=classifier_dict)

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def get_classifier(self, classifier_dict):
        self.classifier = None
        self.SM = None
        self.CEL = None
        if classifier_dict is not None:
            if self.use_sig_for_classifier:
                classifier_dict['input_size'] += self.sig_depth
            self.classifier = FFNN(**classifier_dict)
            self.SM = torch.nn.Softmax(dim=1)
            self.CEL = torch.nn.CrossEntropyLoss()

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        if self.solver == "euler":
            h = h + delta_t * self.ode_f(
                x=last_X, h=h, tau=tau, tdiff=current_time - tau,
                signature=signature)
        else:
            raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def recreate_data(self, times, time_ptr, X, obs_idx, start_X):
        """
        recreates matrix of all observations
        first dim: which data path
        second dim: which time
        """
        # shape: [batch_size, time_steps+1, dimension]
        data = np.empty(shape=(start_X.shape[0], 1+len(times), start_X.shape[1]))
        data[:] = np.nan
        data[:,0,:] = start_X.detach().cpu().numpy()

        X = X.detach().cpu().numpy()
        for j, time in enumerate(times):
            start = time_ptr[j]
            end = time_ptr[j + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            data[i_obs, j+1, :] = X_obs
        times_new = np.concatenate(([0], times), axis=0)

        return times_new, data

    def get_signature(self, times, time_ptr, X, obs_idx, start_X):
        """
        Input: See forward
        Returns: signature of paths as nested list
        """
        # reconstructing the data, shape: [batch_size, time_steps+1, dim]
        times_new, data = self.recreate_data(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            start_X=start_X)

        # list of list of list, shape: [batch_size, obs_dates[j], sig_length]
        signature = []
        for j in range(data.shape[0]):  # iterate over batch
            data_j = data[j, :, :]
            observed_j = []
            for i in range(data_j.shape[0]):
                # if the current batch-sample has an observation at the current
                #   time, add it to the list of observations
                if not np.all(np.isnan(data_j[i])):
                    observed_j += [i]
            data_j = data_j[observed_j, :]

            # replace no observations with last observation
            for i in range(1, data_j.shape[0]):
                # # OLD VERSION (SLOW)
                # for k in range(data_j.shape[1]):
                #     if np.isnan(data_j[i, k]):
                #         data_j[i, k] = data_j[i-1, k]
                ks = np.isnan(data_j[i, :])
                data_j[i, ks] = data_j[i-1, ks]

            times_j = times_new[observed_j].reshape(-1, 1)
            # add times to data for signature call
            path_j = np.concatenate((times_j, data_j), axis=1)
            # the following computes the signatures of all partial paths, from
            #   start to each point of the path
            signature.append(sig.sig(path_j, self.level, 2))

        return signature

    def get_Xy_reg(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_path=False, until_T=False,
            M=None, start_M=None, which_loss=None, dim_to=None,
            predict_labels=None, return_classifier_out=False,
            return_at_last_obs=False):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """

        linreg_X = []
        linreg_y = []

        if which_loss is None:
            which_loss = self.which_loss

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = data_dim
        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, data_dim)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1)
        current_time = 0.0
        loss = 0
        c_sig = None

        if self.input_sig:
            if self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X_obs_impute,
                    obs_idx=obs_idx, start_X=start_X)
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    start_X=start_X)

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float()

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
        else:
            start_M = None

        h = self.encoder_map(
            start_X, mask=start_M, sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)))

        if return_path:
            path_t = [0]
            path_h = [h]
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                    current_time_nb = int(round(current_time / delta_t))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)

            # Reached an observation - only update those elements of the batch,
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end])
                else:
                    M_obs = M[start:end]
            else:
                M_obs = None

            # update signature
            if self.input_sig:
                for j in i_obs:
                    current_sig[j, :] = signature[j][current_sig_nb[j]]
                current_sig_nb[i_obs] += 1
                c_sig = torch.from_numpy(current_sig).float()

            # Using RNNCell to update h. Also updating loss, tau and last_X
            h_bj = h.clone()
            X_obs_impute = X_obs
            temp = h.clone()
            if self.masked:
                # TODO: imputation does not work with OLS -> therefore set to 0
                #  for non-observed coordinates
                X_obs_impute = X_obs * M_obs
            c_sig_iobs = None
            if self.input_sig:
                c_sig_iobs = c_sig[i_obs]
            temp[i_obs.long()] = self.encoder_map(
                X_obs_impute, mask=M_obs, sig=c_sig_iobs, h=h[i_obs])
            h = temp
            h_aj = h.clone()

            # update h and sig at last observation
            h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
            sig_at_last_obs = c_sig

            for ii, o in enumerate(i_obs.long()):
                linreg_X.append(h_bj[o].detach().cpu().numpy())
                linreg_X.append(h_aj[o].detach().cpu().numpy())
                target = X_obs[ii, :dim_to].detach().cpu().numpy()
                if self.masked:
                    target[M_obs[ii, :dim_to].detach().cpu().numpy()==0] = np.nan
                linreg_y.append(target)
                linreg_y.append(target)

            # make update of last_X and tau, that is not inplace
            #    (otherwise problems in autograd)
            temp_X = last_X.clone()
            temp_tau = tau.clone()
            temp_X[i_obs.long()] = X_obs_impute
            if self.coord_wise_tau:
                _M = torch.zeros_like(temp_tau)
                _M[i_obs] = M_obs
                temp_tau[_M==1] = obs_time.astype(np.float64)
            else:
                temp_tau[i_obs.long()] = obs_time.astype(np.float64)
            last_X = temp_X
            tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=self.SM(cl_out), target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            if self.input_sig:
                c_sig = torch.from_numpy(current_sig).float()
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            if return_classifier_out:
                return linreg_X, linreg_y, \
                       np.array(path_t), torch.stack(path_h), cl_out
            return linreg_X, linreg_y, \
                   np.array(path_t), torch.stack(path_h)
        else:
            if return_classifier_out and self.classifier is not None:
                return linreg_X, linreg_y, cl_out
            return linreg_X, linreg_y

    def forward(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=False, get_loss=True, until_T=False,
                M=None, start_M=None, which_loss=None, dim_to=None,
                predict_labels=None, return_classifier_out=False,
                return_at_last_obs=False):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """
        if which_loss is None:
            which_loss = self.which_loss

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = data_dim
        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, data_dim).to(
                self.device)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1).to(self.device)
        current_time = 0.0
        loss = 0
        c_sig = None

        if self.input_sig:
            if self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X_obs_impute,
                    obs_idx=obs_idx, start_X=start_X)
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    start_X=start_X)

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float().to(self.device)

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
        else:
            start_M = None

        h = self.encoder_map(
            start_X, mask=start_M, sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)).to(self.device))

        if return_path:
            path_t = [0]
            path_h = [h]
            path_y = [self.readout_map(h)]
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                    current_time_nb = int(round(current_time / delta_t))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

            # Reached an observation - only update those elements of the batch,
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end]).to(self.device)
                else:
                    M_obs = M[start:end]
            else:
                M_obs = None

            # update signature
            if self.input_sig:
                for j in i_obs:
                    current_sig[j, :] = signature[j][current_sig_nb[j]]
                current_sig_nb[i_obs] += 1
                c_sig = torch.from_numpy(current_sig).float().to(self.device)

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.readout_map(h)
            X_obs_impute = X_obs
            temp = h.clone()
            if self.masked:
                X_obs_impute = X_obs * M_obs
            c_sig_iobs = None
            if self.input_sig:
                c_sig_iobs = c_sig[i_obs]
            temp[i_obs.long()] = self.encoder_map(
                X_obs_impute, mask=M_obs, sig=c_sig_iobs, h=h[i_obs])
            h = temp
            Y = self.readout_map(h)

            # update h and sig at last observation
            h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
            sig_at_last_obs = c_sig

            if get_loss:
                loss = loss + LOSS_FUN_DICT[which_loss](
                    X_obs=X_obs[:, :dim_to], Y_obs=Y[i_obs.long(), :dim_to],
                    Y_obs_bj=Y_bj[i_obs.long(), :dim_to],
                    n_obs_ot=n_obs_ot[i_obs.long()], batch_size=batch_size,
                    weight=self.weight, M_obs=M_obs)

            # make update of last_X and tau, that is not inplace
            #    (otherwise problems in autograd)
            temp_X = last_X.clone()
            temp_tau = tau.clone()
            temp_X[i_obs.long()] = X_obs_impute
            if self.coord_wise_tau:
                _M = torch.zeros_like(temp_tau)
                _M[i_obs] = M_obs
                temp_tau[_M==1] = obs_time.astype(np.float64)
            else:
                temp_tau[i_obs.long()] = obs_time.astype(np.float64)
            last_X = temp_X
            tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                path_y.append(Y)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=self.SM(cl_out), target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            if self.input_sig:
                c_sig = torch.from_numpy(current_sig).float().to(self.device)
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            if return_classifier_out:
                return h, loss, np.array(path_t), torch.stack(path_h), \
                       torch.stack(path_y)[:, :, :dim_to], cl_out
            return h, loss, np.array(path_t), torch.stack(path_h), \
                   torch.stack(path_y)[:, :, :dim_to]
        else:
            if return_classifier_out and self.classifier is not None:
                return h, loss, cl_out
            return h, loss

    def evaluate(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, stockmodel, cond_exp_fun_kwargs=None,
                 diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
                 return_paths=False, M=None, true_paths=None, start_M=None,
                 true_mask=None, mult=None):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :param start_M: see forward
        :param true_paths: np.array, shape [batch_size, dimension, time_steps+1]
        :param true_mask: as true_paths, with mask entries
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        dim = start_X.shape[1]
        dim_to = dim
        if mult is not None and mult > 1:
            dim_to = round(dim/mult)

        _, _, path_t, path_h, path_y = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, None,
            return_path=True, get_loss=False, until_T=True, M=M,
            start_M=start_M, dim_to=dim_to)

        if true_paths is None:
            if M is not None:
                M = M.detach().cpu().numpy()[:, :dim_to]
            _, true_path_t, true_path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X.detach().cpu().numpy()[:, :dim_to],
                obs_idx.detach().cpu().numpy(),
                delta_t, T, start_X.detach().cpu().numpy()[:, :dim_to],
                n_obs_ot.detach().cpu().numpy(),
                return_path=True, get_loss=False, M=M, )
        else:
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t)))
            true_path_y = true_paths[:, :dim_to, which_t_ind]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1))
            true_path_t = true_t[which_t_ind]

        if path_y.detach().cpu().numpy().shape == true_path_y.shape:
            eval_metric = diff_fun(path_y.detach().cpu().numpy(), true_path_y)
        else:
            print(path_y.detach().cpu().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_metric, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_metric

    def evaluate_LOB(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_paths=False, predict_times=None,
            true_predict_vals=None, true_predict_labels=None, true_samples=None,
            normalizing_mean=0., normalizing_std=1., eval_predict_steps=None,
            thresholds=None, predict_labels=None,
            coord_to_compare=(0,), class_report=False):
        """
        evaluate the model at its current training state for the LOB dataset

        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param return_paths: bool, whether to return also the paths
        :param predict_times: np.array with the times at which each sample
                should be predicted
        :param true_predict_vals: np.array with the true values that should be
                predicted at the predict_times
        :param true_predict_labels: np.array, the correct labels at
                predict_times
        :param true_samples: np.array, the true samples, needed to compute
                predicted labels
        :param normalizing_mean: float, the mean with which the price data was
                normalized
        :param normalizing_std: float, the std with which the price data was
                normalized
        :param eval_predict_steps: list of int, the amount of steps ahead at
                which to predict
        :param thresholds: list of float, the labelling thresholds for each
                entry of eval_predict_steps
        :param predict_labels: as true_predict_labels, but as torch.tensor with
                classes in {0,1,2}
        :param coord_to_compare: list or None, the coordinates on which the
                output and input are compared, applied to the inner dimension of
                the time series, e.g. use [0] to compare on the midprice only
        :param class_report: bool, whether to print the classification report
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        bs = start_X.shape[0]
        dim = start_X.shape[1]
        if coord_to_compare is None:
            coord_to_compare = np.arange(dim)

        _, _, path_t, path_h, path_y, cl_out = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=False, until_T=True, M=None,
            start_M=None, dim_to=None, predict_labels=predict_labels,
            return_classifier_out=True)

        path_y = path_y.detach().cpu().numpy()
        predicted_vals = np.zeros_like(true_predict_vals)
        for i in range(bs):
            t = predict_times[i][0]
            t_ind = np.argmin(np.abs(path_t - t))
            predicted_vals[i, :, 0] = path_y[t_ind, i, :]

        eval_metric = np.nanmean(
            (predicted_vals[:, coord_to_compare, 0] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        ref_eval_metric = np.nanmean(
            (true_samples[:, coord_to_compare, -1] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        f1_scores = None
        predicted_labels = None
        if true_samples is not None and true_predict_labels is not None:
            predicted_labels = np.zeros(bs)
            if cl_out is not None:
                class_probs = self.SM(cl_out).detach().cpu().numpy()
                classes = np.argmax(class_probs, axis=1) - 1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], classes,
                    average="weighted")
                predicted_labels = classes
            else:
                # TODO: this computes the labels incorrectly, since the shift by
                #  X_0 is missing -> results should not be trusted, better to
                #  use classifier
                m_minus = np.mean(
                    true_samples[:, 0, -eval_predict_steps[0]:] *
                    normalizing_std + normalizing_mean, axis=1)
                m_plus = predicted_vals[:, 0, 0]*normalizing_std + \
                         normalizing_mean
                pctc = (m_plus - m_minus) / m_minus
                predicted_labels[pctc > thresholds[0]] = 1
                predicted_labels[pctc < -thresholds[0]] = -1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], predicted_labels,
                    average="weighted")
            if class_report:
                print("eval-mse: {:.5f}".format(eval_metric))
                print("f1-score: {:.5f}".format(f1_scores))
                print("classification report \n",
                      sklearn.metrics.classification_report(
                          true_predict_labels[:, 0], predicted_labels,))

        if return_paths:
            return eval_metric, ref_eval_metric, f1_scores, path_t, path_y, \
                   predicted_vals[:, :, 0], predicted_labels
        else:
            return eval_metric, f1_scores

    def get_pred(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, M=None, start_M=None, which_loss=None, dim_to=None):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :param start_M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        h, loss, path_t, path_h, path_y = self.forward(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
            return_path=True, get_loss=True, until_T=True, M=M,
            start_M=start_M, which_loss=which_loss, dim_to=dim_to)
        return {'pred': path_y, 'pred_t': path_t, 'loss': loss}

    def forward_classifier(self, x, y):
        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        cl_loss = None
        if self.classifier is not None:
            cl_out = self.classifier(x)
            cl_loss = self.CEL(input=self.SM(cl_out), target=y)
        return cl_loss, cl_out


class NJmodel(NJODE):
    """
    Neural Jump model without an ODE, i.e. directly learning the Doob-Dynkin
    function
    """
    def __init__(
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param options: kwargs, used:
                - "classifier_nn"
                - "options" with arg a dict passed
                    from train.train (kwords: 'which_loss', 'residual_enc_dec',
                    'residual_enc'
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode', 'enc_input_t' are used)
        """
        super().__init__(
            input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias, dropout_rate, solver,
            weight, weight_decay, **options)

        self.enc_input_t = True

        t_size = 2
        if self.coord_wise_tau:
            t_size = 2*input_size

        self.ode_f = None
        self.encoder_map = FFNN(
            input_size=input_size, output_size=output_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc,
            input_sig=self.input_sig, sig_depth=self.sig_depth,
            input_t=self.enc_input_t, t_size=t_size)
        self.readout_map = torch.nn.Identity()

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        current_time += delta_t
        next_h = self.encoder_map(
            last_X, mask=torch.zeros_like(last_X), sig=signature, h=h,
            t=torch.cat((tau, current_time - tau), dim=1))

        return next_h, current_time


