import jax
import jax.numpy as jnp                # JAX NumPy
from jax import custom_vjp
import numpy as np                     # Ordinary NumPy
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import parameter_properties
from typing import Any
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class


@dataclass
@register_pytree_node_class
class LinearOperator(object):
    matrix: Any
    is_non_singular: Any = None
    is_self_adjoint: Any = None
    is_positive_definite: Any = None
    is_square: Any = None
    name: str = ''
    
    @property
    def dtype(self):
        return self.matrix.dtype
    
    @property
    def shape(self):
        return self.matrix.shape
    
    @property
    def domain_dimension(self):
        return self.matrix.shape[-1]
    
    def domain_dimension_tensor(self):
        return self.matrix.shape[-1]
    
    @property
    def batch_shape(self):
        return self.matrix.shape[:-2]
    
    @property
    def range_dimension(self):
        return self.matrix.shape[-2]
    
    @property
    def graph_parents(self):
        return []
    
    def tree_flatten(self):
        children = (self.matrix,)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def matmul(self, x, adjoint=False, adjoint_arg=False):
        matrix = jnp.swapaxes(self.matrix, -1, -2) if adjoint else self.matrix
        x = jnp.swapaxes(x, -1, -2) if adjoint_arg else x
        return jnp.matmul(matrix, x)
    
    def solve(self, x, adjoint=False, adjoint_arg=False):
        matrix = jnp.swapaxes(self.matrix, -1, -2) if adjoint else self.matrix
        x = jnp.swapaxes(x, -1, -2) if adjoint_arg else x
        return jnp.linalg.solve(matrix, x)
    
    def log_abs_determinant(self):
        return jnp.linalg.slogdet(self.matrix)[1]

class InverseOperator(LinearOperator):
    def solve(self, x, adjoint=False, adjoint_arg=False):
        matrix = jnp.swapaxes(self.matrix, -1, -2) if adjoint else self.matrix
        x = jnp.swapaxes(x, -1, -2) if adjoint_arg else x
        return jnp.matmul(matrix, x)
    
    def matmul                     (self, x, adjoint=False, adjoint_arg=False):
        matrix = jnp.swapaxes(self.matrix, -1, -2) if adjoint else self.matrix
        x = jnp.swapaxes(x, -1, -2) if adjoint_arg else x
        return jnp.linalg.solve(matrix, x)
    
    
class MatrixNormal(tfd.MatrixNormalLinearOperator):
    def __init__(self, loc, scale_row, scale_column):
        tfd.MatrixNormalLinearOperator.__init__(self, loc, LinearOperator(scale_row), LinearOperator(scale_column))

class Wishart(tfd.TransformedDistribution):
    def __init__(self, df, V, inv_V=False):
        bij = tfb.Chain([tfb.CholeskyOuterProduct()])
        SinvTriL = bij.inverse(V)
        if inverse_param:
            SinvTriL = tfb.CholeskyToInvCholesky().forward(SinvTriL)
        dist = tfd.WishartTriL(df, SinvTriL, input_output_cholesky=True)
        tfd.TransformedDistribution.__init__(self, dist, bij)
        
class InverseWishart(tfd.TransformedDistribution):
    def __init__(self, df, S):
        bij = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.CholeskyToInvCholesky()])
        SinvTriL = bij.inverse(S)
        dist = tfd.WishartTriL(df, SinvTriL, input_output_cholesky=True)
        tfd.TransformedDistribution.__init__(self, dist, bij)
        
class InverseWishartTriL(tfd.TransformedDistribution):
    def __init__(self, df, STriL, input_output_cholesky=True):
        assert input_output_cholesky
        bij = tfb.Chain([tfb.CholeskyToInvCholesky()])
        SinvTriL = bij.inverse(S)
        dist = tfd.WishartTriL(df, SinvTriL, input_output_cholesky=True)
        tfd.TransformedDistribution.__init__(self, dist, bij)

class NormalWishart(tfd.Distribution):
    def __init__(self, loc, df, V, lam, inv_V=False):
        self.w = Wishart(df, V, inv_V=inv_V)
        self.loc = loc
        self.lam = lam
        tfd.Distribution.__init__(self)
        
    def log_prob(self, X, L):
        return self.w.log_prob(L) + tfd.MultivariateNormalFullCovariance(self.loc, InverseOperator(L * self.lam)).log_prob(X)
    
    def sample(self, sample_shape=(), seed=None):
        xseed, covseed = jax.random.split(seed)
        L = self.w.sample(sample_shape, seed=covseed)
        return tfd.MultivariateNormalFullCovariance(self.loc, InverseOperator(L * self.lam)).sample(seed=xseed), Sigma        

class NormalInverseWishart(tfd.Distribution):
    def __init__(self, loc, df, S, lam):
        self.iw = InverseWishart(df, S)
        self.loc = loc
        self.lam = lam
        
    def log_prob(self, X, Sigma):
        return self.iw.log_prob(Sigma) + tfd.MultivariateNormalFullCovariance(self.loc, Sigma / self.lam).log_prob(X)
    
    def sample(self, sample_shape=(), seed=None):
        xseed, covseed = jax.random.split(seed)
        Sigma = self.iw.sample(sample_shape, seed=covseed)
        return tfd.MultivariateNormalFullCovariance(self.loc, Sigma / self.lam).sample(seed=xseed), Sigma

class MatrixNormalInverseWishart(tfd.Distribution):
    def __init__(self, loc, df, S, V):
        self.iw = InverseWishart(df, S)
        self.loc = loc
        self.V = V
        
    def log_prob(self, X, Sigma=None):
        if Sigma is None:
            X, Sigma = X[..., :-self.S.shape[-1]], X[..., :-self.S.shape[-1]:]
        return self.iw.log_prob(Sigma) + MatrixNormal(self.loc, Sigma, self.V).log_prob(X)
    
    def sample(self, sample_shape=(), seed=None):
        xseed, covseed = jax.random.split(seed)
        Sigma = self.iw.sample(sample_shape, seed=covseed)
        return MatrixNormal(self.loc, Sigma, self.V).sample(seed=xseed), Sigma
    
