"""Operator table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, - out_grad * (lhs / rhs) / rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.swapaxes(a, -2, -1)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple([1] * (len(self.shape) - len(a.shape)) \
            + list(a.shape))
        return array_api.broadcast_to(
            a.compact().reshape(new_shape), self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes = []
        input_shape = node.inputs[0].shape
        input_shape_padded = [1] * (len(out_grad.shape) - len(input_shape)) \
                             + list(input_shape)

        for axis, dim in enumerate(input_shape_padded):
            if dim != out_grad.shape[axis]:
                axes.append(axis)

        in_grad = summation(out_grad, tuple(axes))
        return reshape(in_grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if type(axes) is int:
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = a
        axes = self.axes if self.axes is not None \
            else (range(len(a.shape)))
        axes = sorted(list(axes), reverse=True)
        for axis in axes:
            out = array_api.sum(out, axis)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)
        axes = self.axes if self.axes is not None else range(len(shape))
        for axis in axes:
            shape[axis] = 1
        return broadcast_to(
            reshape(out_grad, shape),
            node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_axes = [axis for axis in range(len(node.shape) - len(a.shape))]
        b_axes = [axis for axis in range(len(node.shape) - len(b.shape))]
        return summation(matmul(out_grad, transpose(b)), tuple(a_axes)), \
               summation(matmul(transpose(a), out_grad), tuple(b_axes))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return divide(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return multiply(out_grad, exp(a))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data().numpy()
        in_grad = out_grad.realize_cached_data().numpy()
        in_grad[a < 0] = 0
        in_grad = Tensor(in_grad, device=out_grad.device, dtype=out_grad.dtype)
        return in_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        # if type(axes) is int:
        #     axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return array_api.log(
            array_api.sum(
                array_api.exp(
                    Z - array_api.broadcast_to(
                        array_api.amax(Z, self.axes, keepdims=True),
                        Z.shape,
                    )),
                self.axes)
        ) + array_api.amax(Z, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        a_max = array_api.broadcast_to(
            array_api.amax(a.cached_data, self.axes, keepdims=True),
            a.shape,
        )
        a_max = Tensor(a_max, dtype=out_grad.dtype, device=out_grad.device)
        x = a - a_max

        sum_shape = list(x.shape)
        axes = self.axes if self.axes is not None else range(len(x.shape))
        for axis in axes:
            sum_shape[axis] = 1
        transfer = lambda t: broadcast_to(reshape(t, sum_shape), x.shape)

        exp_x = exp(x)
        sum_exp_x = summation(exp_x, self.axes)
        in_grad = exp_x / transfer(sum_exp_x) * transfer(out_grad)
        return in_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return multiply(out_grad, 1 - power_scalar(tanh(a), 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape_prod = 1
        out_shape = [len(args)]

        for size in args[0].shape:
            shape_prod *= size
            out_shape.append(size)

        out = array_api.empty(
            (len(args), shape_prod), dtype=args[0].dtype, device=args[0].device)

        for i, arg in enumerate(args):
            flatted_arg = arg.compact().reshape((shape_prod,))
            out[i, :] = flatted_arg

        out_axes = tuple(list(range(1, self.axis + 1)) + [0] \
                         + list(range(self.axis + 1, len(args[0].shape) + 1)))
        out = out.compact().reshape(out_shape).permute(out_axes).compact()

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        in_shape = A.shape
        out_shape = []
        shape_prod = 1

        for axis, size in enumerate(in_shape):
            if axis != self.axis:
                shape_prod *= size
                out_shape.append(size)

        permuted_axes = [self.axis] + list(range(0, self.axis)) \
            + list(range(self.axis + 1, len(A.shape)))
        flatted_shape = (in_shape[self.axis], shape_prod)
        A_flatted = A.permute(permuted_axes).compact().reshape(flatted_shape)

        outs = []
        out_shape = tuple(out_shape)
        for i in range(A_flatted.shape[0]):
            outs.append(A_flatted[i, :].compact().reshape(out_shape))

        return outs
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out_shape = list(a.shape)
        for axis in self.axes:
            if axis < len(out_shape):
                out_shape[axis] = out_shape[axis] * (self.dilation + 1)

        out_shape = tuple(out_shape)
        out = array_api.full(
            shape=out_shape, fill_value=0, 
            dtype=a.dtype, device=a.device)

        slices = []
        for axis, size in enumerate(out_shape):
            if axis in self.axes:
                slices.append(slice(0, size, self.dilation + 1))
            else:
                slices.append(slice(0, size, 1))

        slices = tuple(slices)
        out[slices] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = []

        for axis, size in enumerate(a.shape):
            if axis in self.axes:
                slices.append(slice(0,size,self.dilation + 1))
            else:
                slices.append(slice(0,size,1))

        slices = tuple(slices)
        return a[slices]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        X, W = A, B
        X_padded = X.pad((
            (0, 0), (self.padding, self.padding), 
            (self.padding, self.padding), (0, 0),
        )).compact()

        N, H_in, W_in, C_in = X_padded.shape
        K1, K2, _, C_out = W.shape
        Ns, Hs, Ws, Cs = X_padded.strides

        H_out = (H_in - (K1 - 1) - 1) // self.stride + 1
        W_out = (W_in - (K2 - 1) - 1) // self.stride + 1
        
        outer_dim = N * H_out * W_out
        inner_dim = K1 * K2 * C_in

        X_reshaped = X_padded.as_strided(
            shape=(N, H_out, W_out, K1, K2, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).compact().reshape(new_shape=(outer_dim,inner_dim))
        W_reshaped = W.compact().reshape(new_shape=(inner_dim, C_out))

        out = X_reshaped @ W_reshaped
        out = out.compact().reshape(new_shape=(N, H_out, W_out, C_out))

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        N, H_in, W_in, C_in = X.shape
        K1, K2, C_in, C_out = W.shape
        N, H_out, W_out, C_out = out_grad.shape
        H_padding = ((H_in - 1) - H_out * self.stride + K1) // 2

        W_flipped = flip(W, axes=(0, 1))  # K1, K2, C_in, C_out
        W_transposed = transpose(W_flipped, axes=(2, 3))  # K1, K2, C_out, C_in
        out_grad_dilated = dilate(out_grad, axes=(1, 2), 
            dilation=self.stride - 1)  # N, H_out, W_out, C_out
        X_grad = conv(out_grad_dilated, W_transposed, 
            stride=1, padding=H_padding)  # N, H_in, W_in, C_in

        X_transposed = transpose(X, axes=(0, 3))  # C_in, H_in, W_in, N
        out_grad_transposed = transpose(
            transpose(out_grad_dilated, axes=(0, 2)), 
            axes=(0, 1))  # H_out, W_out, N, C_out
        W_grad = conv(X_transposed, out_grad_transposed, 
            stride=1, padding=self.padding)  # C_in, K1, K2, C_out
        W_grad = transpose(
            transpose(W_grad, (0, 2)), (0, 1))  # K1, K2, C_in, C_out

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


def norm(x, p=2, dim=1):
    return summation(x ** p, axes=dim) ** (1 / p)


def cosine_similarity(x1, x2, dim=1):
    out = summation(x1 * x2, axes=dim)
    x1_norm = norm(x1, dim=dim)
    x2_norm = norm(x2, dim=dim)
    return out / (x1_norm * x2_norm)


def pairwise_distance(x1, x2, p=2, dim=1):
    return norm(x1 - x2, p=p, dim=dim)


class Cholesky(TensorOp):
    def __init__(self):
        pass

    def compute(self, A: NDArray):
        """
        Cholesky decomposition, mirroring LAPACK's DPOTF2
        """
        ### BEGIN YOUR SOLUTION
        import numpy.linalg as la
        return NDArray(la.cholesky(A.numpy()), device=A.device)
        ### END YOUR SOLUTION

    @staticmethod
    def _Phi(A):
        """
        Return lower-triangle of matrix and halve the diagonal
        """
        import numpy as np
        A = np.tril(A)
        A[np.diag_indices_from(A)] *= 0.5
        return A

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Reverse-mode differentiation through the Cholesky decomposition
        """
        ### BEGIN YOUR SOLUTION
        import numpy.linalg as la

        L_bar = out_grad.realize_cached_data().numpy()
        L = node.realize_cached_data().numpy()

        P = self._Phi(L.T @ L_bar)
        L_inv = la.inv(L)
        A_bar = self._Phi(L_inv.T @ (P + P.T) @ L_inv)
        A_bar = (A_bar + A_bar.T) / 2

        return Tensor(A_bar, device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def cholesky(a):
    return Cholesky()(a)
