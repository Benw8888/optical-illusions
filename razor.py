import jax
import jax.numpy as np

from operator import mul
from functools import reduce


def shape_to_size(shape):
    return reduce(mul, shape, 1)

class Trace:
    def __init__(self, buf=None):
        self.buf = buf
        self.i = 0
        self.log_p = 0.

    def log_factor(self, f):
        self.log_p += f

    def gaussian_mixture_sample(self, shape, p1, mu1, sigma1, mu2, p2, sigma2):
        pt = p1 + p2
        p1 = p1 / pt
        p2 = p2 / pt
        size = shape_to_size(shape)
        if self.buf is None:
            self.i += size
            return np.zeros(shape)
        samples = self.buf[self.i : self.i + size].reshape(shape)
        self.i += size
        self.log_p = self.log_p + np.log(
            p1 * jax.scipy.stats.norm.pdf(samples, mu1, sigma1).sum() +
            p2 * jax.scipy.stats.norm.pdf(samples, mu2, sigma2).sum()
        )
        return samples

    
    def gaussian_mixture_observe(self, p1, mu1, sigma1, p2, mu2, sigma2, Z):
        if self.buf is None:
            return
        self.log_p = self.log_p + np.log(
            p1 * jax.scipy.stats.norm.pdf(Z, mu1, sigma1).sum() +
            p2 * jax.scipy.stats.norm.pdf(Z, mu2, sigma2).sum()
        )

    def gaussian_sample(self, shape, mu=0., sigma=1.):
        size = shape_to_size(shape)
        if self.buf is None:
            self.i += size
            return np.zeros(shape)
        size = shape_to_size(shape)
        samples = self.buf[self.i : self.i + size].reshape(shape)
        self.i += size
        self.log_p = self.log_p + jax.scipy.stats.norm.logpdf(samples, mu, sigma).sum()
        return samples

    def gaussian_observe(self, mu, sigma, Z):
        if self.buf is None:
            return
        self.log_p = self.log_p + jax.scipy.stats.norm.logpdf(Z, mu, sigma).sum()
        
    def gaussian_observe_fuzzy(self, mu1, mu2, fraction_2D, sigma, Z):
        # probability that Z is generated from a gaussian noise applied to the mean; 
        #the mean has a p chance of being mu2 and a 1-p chance of being mu1
        p = fraction_2D
        if self.buf is None:
            return
        # the probability is p*(gaussian(Z,mu2)) + (1-p)(gaussian(Z,mu1)). Then we apply logs
        self.log_p = self.log_p + jax.numpy.logaddexp(np.log(p)+jax.scipy.stats.norm.logpdf(Z, mu2, sigma).sum(), np.log(1-p)+jax.scipy.stats.norm.logpdf(Z, mu1, sigma).sum())
        
    def uniform_sample(self, shape):
        '''
        Returns a uniform sample in the range (0, 1), backed by a Gaussian(!).
        '''
        return jax.scipy.stats.norm.cdf(self.gaussian_sample(shape))
    
#     def binary_sample(self, shape, p=0.5):
#         # p is the probability that 1 is returned
#         print("ERROR! don't use, non differentiable")
#         size = shape_to_size(shape)
#         if self.buf is None:
#             self.i += size
#             return np.zeros(shape)
#         size = shape_to_size(shape)
#         samples = self.buf[self.i : self.i + size].reshape(shape) < jax.scipy.stats.norm.cdf(p)
#         self.i += size
#         self.log_p = self.log_p + (np.sum(samples))np.log(p) + (size - np.sum(samples))*np.log(1-p) #jax.scipy.stats.norm.logpdf(samples, mu, sigma).sum()
#         return samples

class Model():
    def __init__(self, *aux):
        t = Trace()
        self.forward(t, *aux)
        self.N = t.i
        self.grad_log_p = jax.grad(self.log_p, argnums=0)

    def log_p(self, x, *aux):
        assert x.shape == (self.N,)
        t = Trace(x)
        self.forward(t, *aux)
        assert t.i == self.N
        return t.log_p

    def forward(self, t):
        pass
    
    def unpack(self, sample, *aux):
        t = Trace(sample)
        return self.forward(t, *aux)


def hmc_step(m, q, key, L=100, L_post=0, eps=1e-5, *aux):
    dU = m.grad_log_p

    def hmc_iter(pqae):
        p, q, aux, eps = pqae
        print("p, q, aux, eps: ", p, q, aux, eps)
        p = p + dU(q, *aux) * eps / 2
        print("p transformed after dU: ", p)
        q = q + p * eps
        p = p + dU(q, *aux) * eps / 2
        return (p, q, aux, eps)

    def hmc_iter_rev(pqae):
        p, q, aux, eps = pqae
        return hmc_iter((-p, q, aux, eps))

    p = jax.random.normal(key, (m.N,))
#     p, q = jax.lax.fori_loop(0, L, hmc_iter, (p, q))
    import reversible
    p, q, aux, eps = reversible.my_fori_loop(L, hmc_iter, hmc_iter_rev, (p, q, aux, eps))

# This is equivalent, but slower!
#     for i in range(L):
#         p, q = hmc_iter(i, (p, q))

# This makes things faster!
    if L_post > 0:
        p, q = (jax.lax.stop_gradient(p), jax.lax.stop_gradient(q))
#         p, q = jax.lax.fori_loop(0, L_post, hmc_iter, (p, q))
        p, q, aux, eps = reversible.my_fori_loop(L, hmc_iter, hmc_iter_rev, (p, q, aux, eps))

    p = -p  # superstition?
    return q


def hmc_sample(m, q0, key, N, L, L_post, eps, *aux):
    print("HMC sample N: ", N)
    def get_one_sample(c, i):
        print("get one sample inputs c,i: ", c,i)
        q0, key = c
        print("q0, key: ", q0, key)
        key, subkey = jax.random.split(key)
        print("key, subkey: ", key, subkey)
        q0 = hmc_step(m, q0, subkey, L, L_post, eps, *aux)
        print("q0[0]: ", q0[0])
        return (q0, key), q0
    c, samples = jax.lax.scan(get_one_sample, (q0, key), None, length=N)
    print("samples[0,0]: ", c, samples[0,0])

# This is equivalent, but slower!
#     samples = []
#     for i in range(N):
#         key, subkey = jax.random.split(key)
#         q0 = hmc_step(m, q0, key, L, eps, *aux)
#         samples.append(q0)
#     samples = np.stack(samples)

    return samples
hmc_sample = jax.jit(hmc_sample, static_argnums=(0, 3, 4, 5))
