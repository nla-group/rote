# Copyright (c) 2021, nla group, manchester
# All rights reserved. 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import collections
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

from scipy.stats import norm
from scipy.interpolate import interp1d

# SAX
class SAX:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.breakpoints = None
        self.mean = None
        self.std = None
        self.segment_sizes = None
        self.last_paa_values = None
        self.last_symbols = None
    
    def fit(self, X):
        """Normalize the time series by computing mean and standard deviation.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        self : SAX
            The fitted SAX model.
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X)
        self.std = np.std(X) if np.std(X) > 1e-10 else 1.0
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        return self
    
    def transform(self, X):
        """Compute Piecewise Aggregate Approximation (PAA) and map to symbols.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence representing the time series.
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean) / self.std
        n_samples = len(X)
        if n_samples < self.window_size:
            raise ValueError(f"Input length {n_samples} < window_size {self.window_size}")
        base_size = n_samples // self.window_size
        remainder = n_samples % self.window_size
        self.segment_sizes = [base_size + 1 if i < remainder else base_size for i in range(self.window_size)]
        paa_values = []
        symbols = []
        idx = 0
        for size in self.segment_sizes:
            segment = X_norm[idx:idx + size]
            mean_val = np.mean(segment)
            paa_values.append(mean_val)
            symbol = np.digitize([mean_val], self.breakpoints)[0]
            symbols.append(chr(97 + symbol))
            idx += size
        self.last_paa_values = np.array(paa_values)
        self.last_symbols = np.array(symbols)
        return self.last_symbols
    
    def fit_transform(self, X):
        """Fit the model and transform the time series to symbols in one step.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence representing the time series.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, symbols=None):
        """Reconstruct the time series from symbols or stored PAA values.
        
        Parameters
        ----------
        symbols : list of str, optional
            The symbolic sequence to reconstruct from. If None, uses stored symbols.
        
        Returns
        -------
        recon : ndarray
            The reconstructed time series.
        """
        if symbols is None:
            symbols = self.last_symbols
            paa_values = self.last_paa_values
        else:
            symbols = np.array([ord(s) - 97 for s in symbols])
            breakpoints_ext = np.concatenate([[-np.inf], self.breakpoints, [np.inf]])
            paa_values = np.array([norm.expect(lambda x: x, loc=0, scale=1, 
                                              lb=breakpoints_ext[i], ub=breakpoints_ext[i+1])
                                  for i in symbols])
        recon_norm = []
        for val, size in zip(paa_values, self.segment_sizes):
            recon_norm.extend([val] * size)
        recon = np.array(recon_norm) * self.std + self.mean
        return recon

# SAX-TD
class SAXTD:
    def __init__(self, window_size, alphabet_size, slope_threshold=0.01):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.slope_threshold = slope_threshold
        self.breakpoints = None
        self.mean = None
        self.std = None
        self.segment_sizes = None
        self.last_paa_values = None
        self.last_slopes = None
        self.last_symbols = None
    
    def fit(self, X):
        """Normalize the time series by computing mean and standard deviation.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        self : SAXTD
            The fitted SAX-TD model.
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X)
        self.std = np.std(X) if np.std(X) > 1e-10 else 1.0
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        return self
    
    def transform(self, X):
        """Compute PAA, slopes, and symbols with trend information.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with trend suffixes ('u', 'd', 'f').
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean) / self.std
        n_samples = len(X)
        if n_samples < self.window_size:
            raise ValueError(f"Input length {n_samples} < window_size {self.window_size}")
        base_size = n_samples // self.window_size
        remainder = n_samples % self.window_size
        self.segment_sizes = [base_size + 1 if i < remainder else base_size for i in range(self.window_size)]
        paa_values = []
        slopes = []
        symbols = []
        idx = 0
        for size in self.segment_sizes:
            start = idx
            end = idx + size
            segment = X_norm[start:end]
            mean_val = np.mean(segment)
            slope, _ = np.polyfit(np.arange(len(segment)), segment, 1)
            paa_values.append(mean_val)
            slopes.append(slope)
            mean_symbol = np.digitize([mean_val], self.breakpoints)[0]
            trend = 'u' if slope > self.slope_threshold else ('d' if slope < -self.slope_threshold else 'f')
            symbols.append(f"{chr(97 + mean_symbol)}{trend}")
            idx += size
        self.last_paa_values = np.array(paa_values)
        self.last_slopes = np.array(slopes)
        self.last_symbols = np.array(symbols)
        return self.last_symbols
    
    def fit_transform(self, X):
        """Fit the model and transform the time series to symbols with trends.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with trend suffixes.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, symbols=None):
        """Reconstruct the time series with linear trends from symbols.
        
        Parameters
        ----------
        symbols : list of str, optional
            The symbolic sequence with trend suffixes. If None, uses stored symbols.
        
        Returns
        -------
        recon : ndarray
            The reconstructed time series with linear trends.
        """
        if symbols is None:
            paa_values = self.last_paa_values
            slopes = self.last_slopes
        else:
            paa_values = []
            slopes = []
            for s in symbols:
                mean_symbol = ord(s[0]) - 97
                trend = s[1]
                breakpoints_ext = np.concatenate([[-np.inf], self.breakpoints, [np.inf]])
                mean_val = norm.expect(lambda x: x, loc=0, scale=1, 
                                       lb=breakpoints_ext[mean_symbol], 
                                       ub=breakpoints_ext[mean_symbol + 1])
                slope = (self.slope_threshold if trend == 'u' else 
                         (-self.slope_threshold if trend == 'd' else 0))
                paa_values.append(mean_val)
                slopes.append(slope)
            paa_values = np.array(paa_values)
            slopes = np.array(slopes)
        recon_norm = []
        for val, slope, size in zip(paa_values, slopes, self.segment_sizes):
            segment = val + slope * np.linspace(-size/2, size/2, size)
            recon_norm.extend(segment)
        recon = np.array(recon_norm) * self.std + self.mean
        return recon

# eSAX
class ESAX:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.breakpoints = None
        self.mean = None
        self.std = None
        self.segment_sizes = None
        self.last_min_values = None
        self.last_mean_values = None
        self.last_max_values = None
        self.last_symbols = None
    
    def fit(self, X):
        """Normalize the time series by computing mean and standard deviation.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        self : ESAX
            The fitted eSAX model.
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X)
        self.std = np.std(X) if np.std(X) > 1e-10 else 1.0
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        return self
    
    def transform(self, X):
        """Compute min, mean, max per segment and map to symbols.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with three symbols per segment (min, mean, max).
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean) / self.std
        n_samples = len(X)
        if n_samples < self.window_size:
            raise ValueError(f"Input length {n_samples} < window_size {self.window_size}")
        base_size = n_samples // self.window_size
        remainder = n_samples % self.window_size
        self.segment_sizes = [base_size + 1 if i < remainder else base_size for i in range(self.window_size)]
        min_values = []
        mean_values = []
        max_values = []
        symbols = []
        idx = 0
        for size in self.segment_sizes:
            start = idx
            end = idx + size
            segment = X_norm[start:end]
            min_val = np.min(segment)
            mean_val = np.mean(segment)
            max_val = np.max(segment)
            min_values.append(min_val)
            mean_values.append(mean_val)
            max_values.append(max_val)
            min_symbol = np.digitize([min_val], self.breakpoints)[0]
            mean_symbol = np.digitize([mean_val], self.breakpoints)[0]
            max_symbol = np.digitize([max_val], self.breakpoints)[0]
            symbols.append(f"{chr(97 + min_symbol)}{chr(97 + mean_symbol)}{chr(97 + max_symbol)}")
            idx += size
        self.last_min_values = np.array(min_values)
        self.last_mean_values = np.array(mean_values)
        self.last_max_values = np.array(max_values)
        self.last_symbols = np.array(symbols)
        return self.last_symbols
    
    def fit_transform(self, X):
        """Fit the model and transform the time series to symbols.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with three symbols per segment.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, symbols=None):
        """Reconstruct the time series using quadratic interpolation.
        
        Parameters
        ----------
        symbols : list of str, optional
            The symbolic sequence with three symbols per segment. If None, uses stored values.
        
        Returns
        -------
        recon : ndarray
            The reconstructed time series.
        """
        if symbols is None:
            min_values = self.last_min_values
            mean_values = self.last_mean_values
            max_values = self.last_max_values
        else:
            min_values = []
            mean_values = []
            max_values = []
            for s in symbols:
                breakpoints_ext = np.concatenate([[-np.inf], self.breakpoints, [np.inf]])
                min_val = norm.expect(lambda x: x, loc=0, scale=1, 
                                      lb=breakpoints_ext[ord(s[0]) - 97], 
                                      ub=breakpoints_ext[ord(s[0]) - 96])
                mean_val = norm.expect(lambda x: x, loc=0, scale=1, 
                                       lb=breakpoints_ext[ord(s[1]) - 97], 
                                       ub=breakpoints_ext[ord(s[1]) - 96])
                max_val = norm.expect(lambda x: x, loc=0, scale=1, 
                                      lb=breakpoints_ext[ord(s[2]) - 97], 
                                      ub=breakpoints_ext[ord(s[2]) - 96])
                min_values.append(min_val)
                mean_values.append(mean_val)
                max_values.append(max_val)
            min_values = np.array(min_values)
            mean_values = np.array(mean_values)
            max_values = np.array(max_values)
        recon_norm = []
        for min_val, mean_val, max_val, size in zip(min_values, mean_values, max_values, self.segment_sizes):
            x = [0, size/2, size-1]
            y = [min_val, mean_val, max_val]
            f = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
            segment = f(np.arange(size))
            recon_norm.extend(segment)
        recon = np.array(recon_norm) * self.std + self.mean
        return recon

# mSAX
class MSAX:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.breakpoints = None
        self.means = None
        self.stds = None
        self.segment_sizes = None
        self.last_paa_values = None
        self.last_symbols = None
    
    def fit(self, X):
        """Normalize multivariate time series per dimension.
        
        Parameters
        ----------
        X : array-like
            The input multivariate time series.
        
        Returns
        -------
        self : MSAX
            The fitted mSAX model.
        """
        X = np.asarray(X, dtype=np.float64)
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds < 1e-10] = 1.0
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        return self
    
    def transform(self, X):
        """Compute PAA per dimension and map to symbols.
        
        Parameters
        ----------
        X : array-like
            The input multivariate time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with one symbol per dimension per segment.
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.means) / self.stds
        n_samples, n_dims = X.shape
        if n_samples < self.window_size:
            raise ValueError(f"Input length {n_samples} < window_size {self.window_size}")
        base_size = n_samples // self.window_size
        remainder = n_samples % self.window_size
        self.segment_sizes = [base_size + 1 if i < remainder else base_size for i in range(self.window_size)]
        paa_values = []
        symbols = []
        idx = 0
        for size in self.segment_sizes:
            start = idx
            end = idx + size
            segment = X_norm[start:end, :]
            mean_vals = np.mean(segment, axis=0)
            paa_values.append(mean_vals)
            dim_symbols = [np.digitize([val], self.breakpoints)[0] for val in mean_vals]
            symbols.append("".join(chr(97 + s) for s in dim_symbols))
            idx += size
        self.last_paa_values = np.array(paa_values)
        self.last_symbols = np.array(symbols)
        return self.last_symbols
    
    def fit_transform(self, X):
        """Fit the model and transform the multivariate time series to symbols.
        
        Parameters
        ----------
        X : array-like
            The input multivariate time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence with one symbol per dimension per segment.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, symbols=None):
        """Reconstruct the multivariate time series from symbols.
        
        Parameters
        ----------
        symbols : list of str, optional
            The symbolic sequence with one symbol per dimension. If None, uses stored values.
        
        Returns
        -------
        recon : ndarray
            The reconstructed multivariate time series.
        """
        if symbols is None:
            paa_values = self.last_paa_values
        else:
            paa_values = []
            for s in symbols:
                vals = []
                for c in s:
                    idx = ord(c) - 97
                    breakpoints_ext = np.concatenate([[-np.inf], self.breakpoints, [np.inf]])
                    val = norm.expect(lambda x: x, loc=0, scale=1, 
                                      lb=breakpoints_ext[idx], ub=breakpoints_ext[idx + 1])
                    vals.append(val)
                paa_values.append(vals)
            paa_values = np.array(paa_values)
        n_dims = paa_values.shape[1]
        recon_norm = np.zeros((sum(self.segment_sizes), n_dims))
        idx = 0
        for vals, size in zip(paa_values, self.segment_sizes):
            recon_norm[idx:idx + size, :] = vals
            idx += size
        recon = recon_norm * self.stds + self.means
        return recon

# ASAX
class ASAX:
    def __init__(self, n_segments, alphabet_size, min_segment_size=5):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.min_segment_size = min_segment_size
        self.breakpoints = None
        self.mean = None
        self.std = None
        self.segment_bounds = None
        self.segment_sizes = None
        self.last_paa_values = None
        self.last_symbols = None
    
    def fit(self, X):
        """Normalize and compute adaptive segment sizes based on local variance.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        self : ASAX
            The fitted aSAX model.
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X)
        self.std = np.std(X) if np.std(X) > 1e-10 else 1.0
        self.breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        n_samples = len(X)
        X_norm = (X - self.mean) / self.std
        
        # Validate n_segments
        if self.n_segments > n_samples:
            raise ValueError(f"n_segments ({self.n_segments}) cannot exceed n_samples ({n_samples})")
        if self.min_segment_size * self.n_segments > n_samples:
            print(f"Warning: min_segment_size ({self.min_segment_size}) too large for n_samples ({n_samples}). Adjusting.")
        
        # Compute adaptive segment sizes based on local std
        window = max(n_samples // self.n_segments, 1)
        stds = [np.std(X_norm[i:i + window]) if i + window <= n_samples else np.std(X_norm[i:]) 
                for i in range(0, n_samples, window)]
        stds = np.array(stds)
        if len(stds) < self.n_segments:
            stds = np.pad(stds, (0, self.n_segments - len(stds)), mode='edge')
        elif len(stds) > self.n_segments:
            stds = stds[:self.n_segments]
        inv_stds = 1 / (stds + 1e-10)
        total_weight = inv_stds.sum()
        
        # Compute initial segment sizes
        segment_sizes = np.round(inv_stds / total_weight * n_samples).astype(int)
        # Ensure at least 1 point per segment
        segment_sizes = np.clip(segment_sizes, 1, None)
        
        # Adjust to ensure sum equals n_samples
        current_sum = segment_sizes.sum()
        if current_sum != n_samples:
            diff = n_samples - current_sum
            if diff > 0:
                # Add points to segments with highest weights
                sorted_indices = np.argsort(inv_stds)[::-1]
                for i in range(diff):
                    segment_sizes[sorted_indices[i % self.n_segments]] += 1
            elif diff < 0:
                # Remove points from segments with lowest weights, respecting min size
                sorted_indices = np.argsort(inv_stds)
                i = 0
                while diff < 0 and i < self.n_segments:
                    idx = sorted_indices[i % self.n_segments]
                    if segment_sizes[idx] > 1:
                        segment_sizes[idx] -= 1
                        diff += 1
                    i += 1
                # If still not resolved, use equal sizes
                if diff < 0:
                    print("Warning: Falling back to equal segment sizes due to constraints")
                    base_size = n_samples // self.n_segments
                    remainder = n_samples % self.n_segments
                    segment_sizes = np.array([base_size + 1 if i < remainder else base_size 
                                             for i in range(self.n_segments)])
        
        self.segment_sizes = segment_sizes
        self.segment_bounds = np.cumsum([0] + segment_sizes.tolist())
        
        # Validate
        print(f"aSAX segment_sizes: {self.segment_sizes}, Sum: {self.segment_sizes.sum()}")
        print(f"aSAX segment_bounds: {self.segment_bounds}")
        if self.segment_bounds[-1] != n_samples or len(self.segment_sizes) != self.n_segments:
            raise ValueError(f"Invalid segment bounds: {self.segment_bounds[-1]} vs {n_samples}, "
                             f"or wrong number of segments: {len(self.segment_sizes)} vs {self.n_segments}")
        
        return self
    
    def transform(self, X):
        """Compute PAA for adaptive segments and map to symbols.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence for adaptive segments.
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean) / self.std
        paa_values = []
        symbols = []
        for i in range(self.n_segments):
            start = self.segment_bounds[i]
            end = self.segment_bounds[i + 1]
            if start >= end or end > len(X_norm):
                raise ValueError(f"Invalid segment [{start}:{end}] for length {len(X_norm)}")
            segment = X_norm[start:end]
            mean_val = np.mean(segment)
            if np.isnan(mean_val):
                mean_val = 0
            paa_values.append(mean_val)
            symbol = np.digitize([mean_val], self.breakpoints)[0]
            symbols.append(chr(97 + symbol))
        self.last_paa_values = np.array(paa_values)
        self.last_symbols = np.array(symbols)
        return self.last_symbols
    
    def fit_transform(self, X):
        """Fit the model and transform the time series to symbols.
        
        Parameters
        ----------
        X : array-like
            The input time series.
        
        Returns
        -------
        symbols : list of str
            The symbolic sequence for adaptive segments.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, symbols=None):
        """Reconstruct the time series from adaptive segments.
        
        Parameters
        ----------
        symbols : list of str, optional
            The symbolic sequence to reconstruct from. If None, uses stored symbols.
        
        Returns
        -------
        recon : ndarray
            The reconstructed time series.
        """
        if symbols is None:
            paa_values = self.last_paa_values
        else:
            paa_values = []
            for s in symbols:
                idx = ord(s) - 97
                breakpoints_ext = np.concatenate([[-np.inf], self.breakpoints, [np.inf]])
                val = norm.expect(lambda x: x, loc=0, scale=1, 
                                  lb=breakpoints_ext[idx], ub=breakpoints_ext[idx + 1])
                paa_values.append(val)
            paa_values = np.array(paa_values)
        recon_norm = []
        for i, val in enumerate(paa_values):
            segment_size = self.segment_bounds[i + 1] - self.segment_bounds[i]
            recon_norm.extend([val] * segment_size)
        recon = np.array(recon_norm) * self.std + self.mean
        return recon




def symbolsAssign(clusters):
    """ automatically assign symbols to different clusters, start with '!'
        from https://github.com/nla-group/fABBA.
    
    Parameters
    ----------
    clusters - list or pd.Series or array
            the list of clusters.
    ----------
    Return:
    
    symbols(list of string), inverse_hash(dict): repectively for the
    corresponding symbolic sequence and the hashmap for inverse transform.
    
    """
    
    alphabet = ['A','a','B','b','C','c','D','d','E','e',
                'F','f','G','g','H','h','I','i','J','j',
                'K','k','L','l','M','m','N','n','O','o',
                'P','p','Q','q','R','r','S','s','T','t',
                'U','u','V','v','W','w','X','x','Y','y','Z','z']
                
    clusters = pd.Series(clusters)
    N = len(clusters.unique())

    cluster_sort = [0] * N 
    counter = collections.Counter(clusters)
    for ind, el in enumerate(counter.most_common()):
        cluster_sort[ind] = el[0]

    if N >= len(alphabet):
        alphabet = [chr(i+33) for i in range(0, N)]
    else:
        alphabet = alphabet[:N]
    hashm = dict(zip(cluster_sort + alphabet, alphabet + cluster_sort))
    strings = [hashm[i] for i in clusters]
    return strings, hashm




# python implementation for aggregation
def aggregate(data, sorting="2-norm", tol=0.5): # , verbose=1
    """Aggregate points into tolerance-based groups.

    Parameters
    ----------
    data : numpy.ndarray
        Array-like input of shape ``(n_samples, n_features)``.
    sorting : {'2-norm', '1-norm', 'lexi'}, default='2-norm'
        Sorting rule used before aggregation.
    tol : float, default=0.5
        Distance tolerance. A point is assigned to a group when its distance
        from the group's starting point is less than or equal to ``tol``.

    Returns
    -------
    labels : numpy.ndarray
        Group labels after aggregation.
    splist : list
        Starting points for the groups.
    nr_dist : int
        Number of pairwise distance calculations.
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    if sorting == "2-norm": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "1-norm": 
        sort_vals = np.linalg.norm(data, ord=1, axis=1)
        ind = np.argsort(sort_vals)
    else:
        ind = np.lexsort((data[:,1], data[:,0]), axis=0) 
        
    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            clustc = data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            dat = clustc - data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab
            else: # apply early stopping
                if sorting == "2-norm" or sorting == "1-norm":
                    if (sort_vals[j] - sort_vals[sp] > tol):
                        break       
                else:
                    if ((data[j,0] - data[sp,0] == tol) and (data[j,1] > data[sp,1])) or (data[j,0] - data[sp,0] > tol): 
                        break
        splist.append([sp, lab] + [num_group] + list(data[sp,:]) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        lab += 1

    return np.array(labels), splist, nr_dist 



def compress(ts, tol=0.5, max_len=np.inf):
    """
    Approximate a time series using a continuous piecewise linear function.
    Parameters
    ----------
    ts - numpy ndarray
        Time series as input of numpy array
    Returns
    -------
    pieces - numpy array
        Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    """

    start = 0
    end = 1
    pieces = list() # np.empty([0, 3])
    x = np.arange(0, len(ts))
    epsilon =  np.finfo(float).eps

    while end < len(ts):
        inc = ts[end] - ts[start]
        err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
        err = np.inner(err, err)

        if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < max_len):
            (lastinc, lasterr) = (inc, err) 
            end += 1
        else:
            # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
            pieces.append([end-start-1, lastinc, lasterr])
            start = end - 1

    # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    pieces.append([end-start-1, lastinc, lasterr])

    return pieces




class fABBA:
    def __init__ (self, tol=0.1, alpha=0.5,  sorting='2-norm', scl=1, verbose=1, max_len=np.inf):
        """
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        scl - int
            Scale for length, default as 1, means 2d-digitization, otherwise implement 1d-digitization.
        verbose - int
            Control logs print, default as 1, print logs.
        max_len - int
            The max length for each segment, default as np.inf. 
        
        """
        
        self.tol = tol
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        self.alpha = alpha
        self.sorting = sorting
        self.compression_rate = None
        self.digitization_rate = None
        
    

    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - array or list
            Time series.

        alpha - float
            Control tolerence for digitization, default as 0.5.

        string_form - boolean
            Whether to return with string form, default as True.
        """
        series = np.array(series).astype(np.float64)
        pieces = np.array(compress(ts=series, tol=self.tol, max_len=self.max_len))
        strings = self.digitize(pieces[:,0:2])
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(strings)), "to", self.centers.shape[0], "symbols.")  
        # strings = ''.join(strings)
        return strings
    
    

    def digitize(self, pieces, early_stopping=True):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce tol and len/inc scaling parameter scl.
        In this variant, a 'temporary' cluster center is used 
        when assigning pieces to clusters. This temporary cluster
        is the first piece available after appropriate scaling 
        and sorting of all pieces. It is *not* necessarily the 
        mean of all pieces in that cluster and hence the final
        cluster centers, which are just the means, might achieve 
        a smaller within-cluster tol.
        """

        if self.sorting not in ['2-norm', '1-norm', 'lexi']:
            raise ValueError("Please refer to a specific and correct sorting way, namely '2-norm', '1-norm' and 'lexi'")
        
        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[0] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces * np.array([self.scl, 1]) / _std
        labels, self.splist, self.nr_dist = aggregate(npieces, self.sorting, self.alpha) # other two variables are used for experiment
        centers = np.zeros((0,2))
        for c in range(len(self.splist)):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
        self.centers = centers
        strings, self.hashmap = symbolsAssign(labels)
        return strings
    

    
    def inverse_transform(self, strings, start=0):
        pieces = self.inverse_digitize(strings, self.centers, self.hashmap)
        pieces = self.quantize(pieces)
        ts = self.inverse_compress(pieces, start)
        return ts

    

    def inverse_digitize(self, strings, centers, hashmap):
        pieces = np.empty([0,2])
        for p in strings:
            pc = centers[int(hashmap[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]


    
    def quantize(self, pieces):
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        else:
            for p in range(len(pieces)-1):
                corr = round(pieces[p,0]) - pieces[p,0]
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1
            pieces[-1,0] = round(pieces[-1,0],0)
        return pieces

    

    def inverse_compress(self, pieces, start):
        """Modified from ABBA package, please see ABBA package to see guidance."""
        
        ts = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            #print(x)
            y = ts[-1] + x
            ts = ts + y[1:].tolist()

        return ts
    
    
    


class ABBA:
    def __init__ (self, tol=0.1, k_cluster=10, verbose=1, max_len=np.inf):
        """
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        
        k_cluster - int
            Number of symbols used for digitization.
        
        verbose - int
            Control logs print, default as 1, print logs.
        
        max_len - int
            The max length for each segment, default as np.inf. 
        
        """
        
        self.tol = tol
        self.verbose = verbose
        self.max_len = max_len
        self.k_cluster = k_cluster
        self.compression_rate = None
        self.digitization_rate = None
    
        

    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - array or list
            Time series.

        alpha - float
            Control tolerence for digitization, default as 0.5.

        string_form - boolean
            Whether to return with string form, default as True.
        """
        series = np.array(series).astype(np.float64)
        pieces = np.array(compress(ts=series, tol=self.tol, max_len=self.max_len))
        strings = self.digitize(pieces[:,0:2])
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(strings)), "to", self.centers.shape[0], "symbols.")  
        # strings = ''.join(strings)
        return strings
    
    

    def digitize(self, pieces, early_stopping=True):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce tol and len/inc scaling parameter scl.
        In this variant, a 'temporary' cluster center is used 
        when assigning pieces to clusters. This temporary cluster
        is the first piece available after appropriate scaling 
        and sorting of all pieces. It is *not* necessarily the 
        mean of all pieces in that cluster and hence the final
        cluster centers, which are just the means, might achieve 
        a smaller within-cluster tol.
        """

        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[0] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces / _std
        print(npieces.shape)
        kmeans = KMeans(n_clusters=self.k_cluster, random_state=0).fit(npieces)
        labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_*_std
        strings, self.hashmap = symbolsAssign(labels)
        return strings
    

    
    def inverse_transform(self, strings, start=0):
        pieces = self.inverse_digitize(strings, self.centers, self.hashmap)
        pieces = self.quantize(pieces)
        ts = self.inverse_compress(pieces, start)
        return ts

    

    def inverse_digitize(self, strings, centers, hashmap):
        pieces = np.empty([0,2])
        for p in strings:
            pc = centers[int(hashmap[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]


    
    def quantize(self, pieces):
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        else:
            for p in range(len(pieces)-1):
                corr = round(pieces[p,0]) - pieces[p,0]
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1
            pieces[-1,0] = round(pieces[-1,0],0)
        return pieces

    

    def inverse_compress(self, pieces, start):
        """Modified from ABBA package, please see ABBA package to see guidance."""
        
        ts = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            #print(x)
            y = ts[-1] + x
            ts = ts + y[1:].tolist()

        return ts
    
    
