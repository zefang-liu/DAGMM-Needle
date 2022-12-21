"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features,
            device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 1,
                device=device, dtype=dtype, requires_grad=True).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out += ops.broadcast_to(
                self.bias, (X.shape[0], self.out_features))
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        out = ops.summation(ops.logsumexp(logits, axes=(1,)) - ops.summation(
            init.one_hot(
                logits.shape[-1], y, 
                device=logits.device, dtype=logits.dtype) * logits, 
            axes=(1,)
        )) / y.shape[0]
        return out
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(
            dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size = x.shape[0]
            x_mean = ops.summation(x, (0,)) / batch_size
            x_var = ops.summation(
                (x - ops.broadcast_to(x_mean, x.shape)) ** 2, (0,)
            ) / batch_size

            self.running_mean = (1 - self.momentum) * self.running_mean \
                + self.momentum * x_mean.data
            self.running_var = (1 - self.momentum) * self.running_var \
                + self.momentum * x_var.data

            x_mean = ops.broadcast_to(x_mean, x.shape)
            x_var = ops.broadcast_to(x_var, x.shape)
            weight = ops.broadcast_to(self.weight, x.shape)
            bias = ops.broadcast_to(self.bias, x.shape)

            out = (
                (x - x_mean) / (x_var + self.eps) ** 0.5
            ) * weight + bias
        else:
            x_mean = self.running_mean
            x_var = self.running_var
            x_mean = ops.broadcast_to(x_mean, x.shape)
            x_var = ops.broadcast_to(x_var, x.shape)
            weight = ops.broadcast_to(self.weight, x.shape)
            bias = ops.broadcast_to(self.bias, x.shape)

            out = (
                (x - x_mean) / (x_var + self.eps) ** 0.5
            ) * weight.data + bias.data

        return out
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        sum_shape = list(x.shape)
        sum_shape[-1] = 1
        weight = ops.broadcast_to(self.weight, x.shape)
        bias = ops.broadcast_to(self.bias, x.shape)

        x_mean = ops.broadcast_to(
            ops.reshape(
                ops.summation(x, (-1,)) / self.dim, sum_shape
            ), x.shape)
        x_var = ops.broadcast_to(
            ops.reshape(
                ops.summation((x - x_mean) ** 2, (-1,)) / self.dim, sum_shape
            ), x.shape)

        out = (
            (x - x_mean) / (x_var + self.eps) ** 0.5
        ) * weight + bias
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            out = x * init.randb(
                *x.shape, p=(1 - self.p), device=x.device, dtype=x.dtype
            ) / (1 - self.p)
        else:
            out = x
        return out
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
        bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2

        self.weight = Parameter(init.kaiming_uniform(
            in_channels * kernel_size * kernel_size, out_channels, 
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device, dtype=dtype, requires_grad=True))

        if bias:
            self.bias = Parameter(init.rand(
                out_channels, 
                low=-1.0/(in_channels * kernel_size**2)**0.5, 
                high=1.0/(in_channels * kernel_size**2)**0.5,
                device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(
            ops.transpose(x, axes=(1, 3)), 
            axes=(1, 2))  # (N, C, H, W) -> (N, H, W, C)
        out = ops.conv(x, self.weight, 
            stride=self.stride, padding=self.padding)

        if self.bias is not None:
            out += ops.broadcast_to(
                ops.reshape(self.bias, shape=(1, 1, 1, self.out_channels)), 
                shape=out.shape)

        out = ops.transpose(
            ops.transpose(out, axes=(1, 3)), 
            axes=(2, 3))  # (N, H, W, C) -> (N, C, H, W)
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The nonlinearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        k = 1 / hidden_size
        bound = np.sqrt(k)
        
        self.W_ih = Parameter(init.rand(
            input_size, hidden_size, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(
            hidden_size, hidden_size, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True))

        if bias:
            self.bias_ih = Parameter(init.rand(
                hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(
                hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        hidden_size = self.hidden_size

        if h is None:
            h = init.zeros(
                bs, hidden_size, device=self.device, dtype=self.dtype)

        h_out = X @ self.W_ih + h @ self.W_hh 
        if self.bias:
            h_out += ops.broadcast_to(
            self.bias_ih + self.bias_hh, shape=(bs, hidden_size))
        
        if self.nonlinearity == 'tanh':
            h_out = Tanh()(h_out)
        elif self.nonlinearity == 'relu':
            h_out = ReLU()(h_out)

        return h_out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None,
                 dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU nonlinearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The nonlinearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise, the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = []

        for i in range(num_layers):
            if i == 0:
                rnn_cell = RNNCell(input_size, hidden_size, 
                    bias=bias, nonlinearity=nonlinearity, 
                    device=device, dtype=dtype)
            else:
                rnn_cell = RNNCell(hidden_size, hidden_size, 
                    bias=bias, nonlinearity=nonlinearity, 
                    device=device, dtype=dtype)
            self.rnn_cells.append(rnn_cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for
            each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        xs = ops.split(X, axis=0)
        if h0:
            h0s = ops.split(h0, axis=0)
        else:
            h0s = [None] * self.num_layers
        hns = []
        last_outputs = []
        outputs = []

        for i in range(self.num_layers):
            last_outputs = outputs
            outputs = []

            if i == 0:
                for t in range(seq_len):
                    if t == 0:
                        h = self.rnn_cells[i](xs[t], h0s[i])
                    else:
                        h = self.rnn_cells[i](xs[t], h)
                    
                    outputs.append(h)
                    if t == seq_len - 1:
                        hns.append(h)
            else:
                for t in range(seq_len):
                    if t == 0:
                        h = self.rnn_cells[i](last_outputs[t], h0s[i])
                    else:
                        h = self.rnn_cells[i](last_outputs[t], h)
                    
                    outputs.append(h)
                    if t == seq_len - 1:
                        hns.append(h)
        
        output = ops.stack(outputs, axis=0)
        hn = ops.stack(hns, axis=0)
        return output, hn
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        k = 1 / hidden_size
        bound = np.sqrt(k)
        
        self.W_ih = Parameter(init.rand(
            input_size, 4 * hidden_size, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(
            hidden_size, 4 * hidden_size, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True))

        if bias:
            self.bias_ih = Parameter(init.rand(
                4 * hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(
                4 * hidden_size, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        hidden_size = self.hidden_size

        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(
                bs, hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(
                bs, hidden_size, device=self.device, dtype=self.dtype)

        H_out = X @ self.W_ih + h0 @ self.W_hh

        if self.bias:
            H_out += ops.broadcast_to(
            self.bias_ih + self.bias_hh, shape=(bs, 4 * hidden_size))
        
        H_out_splitted = ops.split(H_out, axis=1).tuple()
        i = ops.stack(H_out_splitted[0 * hidden_size:1 * hidden_size], axis=1)
        f = ops.stack(H_out_splitted[1 * hidden_size:2 * hidden_size], axis=1)
        g = ops.stack(H_out_splitted[2 * hidden_size:3 * hidden_size], axis=1)
        o = ops.stack(H_out_splitted[3 * hidden_size:4 * hidden_size], axis=1)

        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = Tanh()(g)
        o = Sigmoid()(o)

        c_out = f * c0 + i * g
        h_out = o * ops.tanh(c_out)

        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = []

        for i in range(num_layers):
            if i == 0:
                lstm_cell = LSTMCell(input_size, hidden_size, 
                    bias=bias, device=device, dtype=dtype)
            else:
                lstm_cell = LSTMCell(hidden_size, hidden_size, 
                    bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(lstm_cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state
                for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state
                for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        xs = ops.split(X, axis=0)

        if h:
            h0, c0 = h
            h0s = ops.split(h0, axis=0)
            c0s = ops.split(c0, axis=0)
            inits = [(h0s[i], c0s[i]) for i in range(self.num_layers)]
        else:
            inits = [None] * self.num_layers

        hns = []
        cns = []
        last_outputs = []
        outputs = []

        for i in range(self.num_layers):
            last_outputs = outputs
            outputs = []

            if i == 0:
                for t in range(seq_len):
                    if t == 0:
                        h, c = self.lstm_cells[i](xs[t], inits[i])
                    else:
                        h, c = self.lstm_cells[i](xs[t], (h, c))
                    
                    outputs.append(h)
                    if t == seq_len - 1:
                        hns.append(h)
                        cns.append(c)
            else:
                for t in range(seq_len):
                    if t == 0:
                        h, c = self.lstm_cells[i](last_outputs[t], inits[i])
                    else:
                        h, c = self.lstm_cells[i](last_outputs[t], (h, c))
                    
                    outputs.append(h)
                    if t == seq_len - 1:
                        hns.append(h)
                        cns.append(c)
        
        output = ops.stack(outputs, axis=0)
        hn = ops.stack(hns, axis=0)
        cn = ops.stack(cns, axis=0)
        return output, (hn, cn)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.randn(
            num_embeddings, embedding_dim, mean=0.0, std=1.0,
            device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_reshaped = x.reshape((seq_len * bs, ))
        x_one_hot = init.one_hot(
            n=self.num_embeddings, i=x_reshaped, 
            device=self.device, dtype=self.dtype)
        output = x_one_hot @ self.weight
        output = output.reshape((seq_len, bs, self.embedding_dim))
        return output
        ### END YOUR SOLUTION
