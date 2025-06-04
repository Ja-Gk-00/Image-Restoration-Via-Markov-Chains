import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numba

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y, neighbors, x_obs):
        pass

class LpLoss(LossFunction):
    def __init__(self, p=1, lambda_data=1.0):
        self.p = p
        self.lambda_data = lambda_data
    def loss(self, y, neighbors, x_obs):
        nbr_term = np.sum(np.abs(neighbors - y) ** self.p)
        data_term = self.lambda_data * np.abs(x_obs - y) ** self.p
        return nbr_term + data_term

class LorentzianLoss(LossFunction):
    def __init__(self, sigma):
        self.sigma = sigma
    def _rho(self, z):
        return np.log(1.0 + 0.5 * (z / self.sigma) ** 2)
    def loss(self, y, neighbors, x_obs):
        return np.sum(self._rho(neighbors - y)) + self._rho(x_obs - y)

class PriorFunction(ABC):
    @abstractmethod
    def prior(self, y, neighbors):
        pass

class SparseGradientPrior(PriorFunction):
    def __init__(self, p=2, eps=1e-3):
        self.p = p
        self.eps = eps
    def prior(self, y, neighbors):
        grads = (neighbors - y) / (neighbors + self.eps)
        return np.sum(np.abs(grads) ** self.p)
    
class PottsPrior(PriorFunction):
    def __init__(self, beta_prior=1.0):
        self.beta_prior = beta_prior

    def prior(self, y_sample, neighbors_of_y_sample):
        if neighbors_of_y_sample.size == 0:
            return 0.0
        return self.beta_prior * np.sum(neighbors_of_y_sample != y_sample)
    
    
class TruncatedQuadraticPrior(PriorFunction):
    def __init__(self, beta_prior=1.0, lambda_sq_penalty=1.0, alpha_threshold=100.0):
        self.beta_prior = beta_prior
        self.lambda_sq_penalty = lambda_sq_penalty
        self.alpha_threshold = alpha_threshold

    def prior(self, y_sample, neighbors_of_y_sample):
        if neighbors_of_y_sample.size == 0:
            return 0.0
        squared_diffs = self.lambda_sq_penalty * (neighbors_of_y_sample - y_sample)**2
        truncated_penalties = np.minimum(squared_diffs, self.alpha_threshold)
        return self.beta_prior * np.sum(truncated_penalties)

class WindowPatternTerm:
    def __init__(self, X, window_size=3, n_components=5):
        if X.ndim == 2:
            X = X[..., None]
        H, W, C = X.shape
        self.ws = window_size
        self.filters = []
        self.variances = []
        pad = window_size // 2
        for c in range(C):
            patches = []
            Xc = np.pad(X[..., c], pad, mode='reflect')
            for i in range(pad, pad + H):
                for j in range(pad, pad + W):
                    w = Xc[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
                    patches.append(w)
            patches = np.stack(patches)
            pca = PCA(n_components=n_components)
            pca.fit(patches)
            self.filters.append(pca.components_)
            self.variances.append(pca.explained_variance_)
    def window_term(self, i, j, c, Y):
        pad = self.ws // 2
        Yc = np.pad(Y[..., c], pad, mode='reflect')
        w = Yc[i:i+2*pad+1, j:j+2*pad+1].flatten()
        return np.sum(self.variances[c] * np.abs(self.filters[c].dot(w)))

class GridMRF:
    def __init__(self, X, loss_fn, prior_fn,
                 lambda_r=1.0, window_term=None, lambda_w=0.0,
                 neighbor_offsets=None, binary_map=False):
        if X.ndim == 2:
            X = X[..., None]
        if binary_map:
            X_thr = (X[..., 0] > 0.5).astype(float)
            self.X = X_thr[..., None]
        else:
            self.X = X.astype(float)
        self.loss_fn = loss_fn
        self.prior_fn = prior_fn
        self.window_term = window_term
        self.lambda_r = lambda_r
        self.lambda_w = lambda_w
        self.offsets = neighbor_offsets or [(-1,0),(1,0),(0,-1),(0,1)]
        self.H, self.W, self.C = self.X.shape
        self._nbr_idx = [
            [[(i+di, j+dj) for di, dj in self.offsets
              if 0 <= i+di < self.H and 0 <= j+dj < self.W]
             for j in range(self.W)]
            for i in range(self.H)
        ]
    def _neighbors(self, Y, i, j, c):
        idxs = self._nbr_idx[i][j]
        return np.array([Y[ni, nj, c] for ni, nj in idxs], dtype=float)
    def energy_pixel(self, y, i, j, c, Y):
        nbr = self._neighbors(Y, i, j, c)
        L = self.loss_fn.loss(y, nbr, self.X[i, j, c])
        R = self.prior_fn.prior(y, nbr) if self.prior_fn else 0.0
        W = self.window_term.window_term(i, j, c, Y) if self.window_term else 0.0
        return L + self.lambda_r * R - self.lambda_w * W
    def total_energy(self, Y):
        return sum(
            self.energy_pixel(Y[i,j,c], i, j, c, Y)
            for c in range(self.C)
            for i in range(self.H)
            for j in range(self.W)
        )
    def total_loss(self, Y):
        return sum(
            self.loss_fn.loss(Y[i,j,c], self._neighbors(Y,i,j,c), self.X[i,j,c])
            for c in range(self.C)
            for i in range(self.H)
            for j in range(self.W)
        )


@numba.njit
def _rho_lorentzian(z, sigma):
    return np.log(1.0 + 0.5 * (z / sigma) ** 2)

@numba.njit
def _compute_lorentzian_loss_channel(Y_channel, X_channel, nbr_indices, sigma):
    H, W = Y_channel.shape
    total_loss = 0.0
    
    for i in range(H):
        for j in range(W):
            y_val = Y_channel[i, j]
            x_obs = X_channel[i, j]
            
            total_loss += _rho_lorentzian(x_obs - y_val, sigma)
            
            nbr_count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_val = Y_channel[ni, nj]
                        total_loss += _rho_lorentzian(neighbor_val - y_val, sigma)
                        nbr_count += 1
    
    return total_loss

@numba.njit
def _sample_neigh_quadratic(y_old, neigh, x_obs, lambda_r, beta_prior, beta_t, alpha=99999.0):
    vmin = np.min(neigh)
    vmax = np.max(neigh)
    cand = np.arange(vmin, vmax + 1)
    m = cand.shape[0]
    energies = np.empty(m)
    for idx in range(m):
        v = cand[idx]
        energies[idx] = lambda_r*(v - x_obs)**2 + beta_prior*np.sum(np.minimum((neigh - v)**2, alpha))
    e0 = energies.min()
    probs = np.exp(-beta_t*(energies - e0))
    probs /= probs.sum()
    r = np.random.rand()
    cum = 0.0
    for idx in range(m):
        cum += probs[idx]
        if r < cum:
            return cand[idx]
    return cand[-1]

@numba.njit
def _sample_neigh_potts(y_old, neigh, x_obs, lambda_r, beta_prior, beta_t):
    vmin = np.min(neigh)
    vmax = np.max(neigh)
    cand = np.arange(vmin, vmax + 1)
    m = cand.shape[0]
    energies = np.empty(m)
    for idx in range(m):
        v = cand[idx]
        energies[idx] = lambda_r*(v - x_obs)**2 + beta_prior*np.sum((neigh != v).astype(np.float32))
    e0 = energies.min()
    probs = np.exp(-beta_t*(energies - e0))
    probs /= probs.sum()
    r = np.random.rand()
    cum = 0.0
    for idx in range(m):
        cum += probs[idx]
        if r < cum:
            return cand[idx]
    return cand[-1]

class GibbsSampler:
    def __init__(self, mrf, num_iter=1000, burn_in=200,
                 verbose=False, betas_T=1.0, beta_prior=1.0, estimate_mode='mean', pior_type_for_optimal='quadratic'):
        self.mrf = mrf
        self.num_iter = num_iter
        self.burn_in = burn_in
        self.verbose = verbose
        if type(betas_T) in (int, float):
            betas_T = [betas_T] * self.num_iter
        if len(betas_T) != self.num_iter:
            raise ValueError("betas must be a scalar or a list of length C")
        self.betas_T = betas_T
        self.beta_prior = beta_prior
        self.estimate_mode = estimate_mode
        self.denoised_ = None
        self.last_sample_ = None
        self.history = {'iter': [], 'loss': [], 'energy': []}
        self.pior_type_for_optimal = pior_type_for_optimal

    def fit(self, shuffle_pixels=False, parallel_channels=False, binary_map=False):
    
        X = self.mrf.X
        if X.ndim == 2:
            X = X[..., None]
        H, W, C = X.shape

        if binary_map:
            candidates = np.array([0, 1], dtype=int)
        else:
            candidates = np.arange(256, dtype=int)

        if parallel_channels:
            def run_channel(c):
                print(f"Starting probing for channel {c}")
                Yc = X[..., c].astype(int)
                accum = np.zeros_like(Yc, dtype=float)
                count = 0
                mrf_c = GridMRF(Yc, self.mrf.loss_fn, self.mrf.prior_fn,
                                self.mrf.lambda_r, None, self.mrf.lambda_w,
                                self.mrf.offsets)
                for t in range(1, self.num_iter+1):
                    coords = [(i,j) for i in range(H) for j in range(W)]
                    if shuffle_pixels:
                        np.random.shuffle(coords)
                    for i,j in coords:
                        Es = np.array([
                            mrf_c.energy_pixel(v,i,j,0,Yc[...,None])
                            for v in range(256)
                        ])
                        logp = -Es; logp -= logp.max()
                        p = np.exp(logp); p /= p.sum()
                        Yc[i,j] = np.random.choice(256, p=p)
                    if t > self.burn_in:
                        accum += Yc; count += 1
                    if self.verbose and t % 10 == 0:
                        loss_val = mrf_c.total_loss(Yc[...,None])
                        energy_val = mrf_c.total_energy(Yc[...,None])
                        print(f"[Channel {c}][Iter {t}] Loss={loss_val:.2f}, Energy={energy_val:.2f}")
                # save last sample
                self.last_sample_ = Yc.copy()
                return c, np.round(accum/max(1,count)).astype(int)

            with ThreadPoolExecutor(max_workers=C) as ex:
                results = [f.result() for f in [ex.submit(run_channel,c) for c in range(C)]]
            den = np.zeros((H,W,C),int)
            for c,avg_c in results:
                den[...,c] = avg_c
            self.denoised_ = den if C>1 else den[...,0]
            return self

        else:
            Y = X.copy().astype(int)
            accum = np.zeros_like(Y, float)
            count = 0
            for t in range(1, self.num_iter+1):
                coords = [(c, i, j) for c in range(C) for i in range(H) for j in range(W)]
                if shuffle_pixels:
                    np.random.shuffle(coords)
                for c, i, j in coords:
                    # use `candidates` instead of range(256)
                    Es = np.array([
                        self.mrf.energy_pixel(v, i, j, c, Y)
                        for v in candidates
                    ])
                    logp = -Es; logp -= logp.max()
                    p = np.exp(logp); p /= p.sum()
                    Y[i, j, c] = np.random.choice(candidates, p=p)

                self.last_sample_ = Y.copy()
                if t > self.burn_in:
                    accum += Y; count += 1
                if self.verbose and t % 10 == 0:
                    loss_val = self.mrf.total_loss(Y)
                    energy_val = self.mrf.total_energy(Y)
                    print(f"[Iter {t}] Loss={loss_val:.2f}, Energy={energy_val:.2f}")
                    self.history['iter'].append(t)
                    self.history['loss'].append(loss_val)
                    self.history['energy'].append(energy_val)

            avg = np.round(accum / max(1, count)).astype(int)
            self.denoised_ = avg if C > 1 else avg[..., 0]
            return self

    def fit_optimised(self, shuffle_pixels=False, parallel_channels=False, binary_map=False):
        
        X = self.mrf.X
        if X.ndim == 2:
            X = X[..., None]
        # If binary_map, treat as single-channel
        if binary_map:
            X = X[..., :1]
        H, W, C = X.shape

        def worker(c):
            Xc = X[..., c]
            mrf_c = GridMRF(Xc, self.mrf.loss_fn, self.mrf.prior_fn,
                            self.mrf.lambda_r, None, self.mrf.lambda_w,
                            self.mrf.offsets)
            # initialize Yc: if binary_map, project observed values to {0,1}
            if binary_map:
                Yc = (np.random.rand(H,W) > 0.5).astype(np.int32)
            else:
                Yc = Xc.astype(int)
            accum_c = np.zeros_like(Yc, float)
            count_c = 0
            for t in range(1, self.num_iter+1):
                coords = [(i,j) for i in range(H) for j in range(W)]
                if shuffle_pixels:
                    np.random.shuffle(coords)
                for i, j in coords:
                    neigh = mrf_c._neighbors(Yc[..., None], i, j, 0)
                    x_obs = mrf_c.X[i, j, 0]
                    if binary_map:
                        cand = np.array([0, 1], dtype=np.int32)
                        m = 2
                        energies = np.empty(m, dtype=np.float64)
                        for idx in range(m):
                            v = cand[idx]
                            if self.pior_type_for_optimal == 'quadratic':
                                sq_diffs = (neigh - v) ** 2
                                truncated = np.minimum(sq_diffs, self.beta_prior)
                                energies[idx] = mrf_c.lambda_r * (v - x_obs) ** 2 + self.beta_prior * np.sum(truncated)
                            elif self.pior_type_for_optimal == 'potts':
                                energies[idx] = mrf_c.lambda_r * (v - x_obs) ** 2 + self.beta_prior * np.sum(neigh != v)
                            else:
                                raise ValueError(f"Unknown prior type: {self.pior_type_for_optimal}")
                        e0 = energies.min()
                        probs = np.exp(-self.betas_T[t-1] * (energies - e0))
                        probs /= probs.sum()
                        r = np.random.rand()
                        cum = 0.0
                        for idx in range(m):
                            cum += probs[idx]
                            if r < cum:
                                Yc[i, j] = cand[idx]
                                break
                    else:
                        if self.pior_type_for_optimal == 'quadratic':
                            Yc[i, j] = _sample_neigh_quadratic(
                                Yc[i, j], neigh, Xc[i, j],
                                mrf_c.lambda_r, self.beta_prior, self.betas_T[t-1])
                        elif self.pior_type_for_optimal == 'potts':
                            Yc[i, j] = _sample_neigh_potts(
                                Yc[i, j], neigh, Xc[i, j],
                                mrf_c.lambda_r, self.beta_prior, self.betas_T[t-1])
                        else:
                            raise ValueError(f"Unknown prior type: {self.pior_type_for_optimal}")
                last_sample_ = Yc.copy()
                if t > self.burn_in:
                    accum_c += Yc
                    count_c += 1
                if self.verbose and t % 10 == 0:
                    l = mrf_c.total_loss(Yc[..., None])
                    e = mrf_c.total_energy(Yc[..., None])
                    print(f"[Channel {c}][Iter {t}] Loss={l:.2f}, Energy={e:.2f}")
            avg_c = np.round(accum_c / max(1, count_c)).astype(int)
            return c, avg_c, last_sample_

        # Only one channel if binary_map
        if C == 1 or binary_map:
            results = [worker(0)]
        else:
            if parallel_channels:
                with ThreadPoolExecutor(max_workers=C) as ex:
                    results = [f.result() for f in [ex.submit(worker, c) for c in range(C)]]
            else:
                results = [worker(c) for c in range(C)]

        den = np.zeros((H, W, C), int)
        last_sample = np.zeros((H, W, C), int)
        for c, avg_c, last_c in results:
            den[..., c] = avg_c
            last_sample[..., c] = last_c
        self.denoised_ = den if C > 1 else den[..., 0]
        self.last_sample_ = last_sample if C > 1 else last_sample[..., 0]
        return self


    def estimate(self):
        if self.denoised_ is None:
            raise RuntimeError("Call fit() first")
        if self.estimate_mode == 'map':
            return self.last_sample_
        return self.denoised_
