
"""
knne_adamenn.py

An implementation of ADAMENN local feature-weight estimation and a simple
kNN-ensemble (KNNE) that uses ADAMENN weights to sample feature subspaces.

This file provides three main classes:

- ADAMENNWeights
    Locally adaptive, query-dependent feature relevance estimator following
    Domeniconi, Peng, Gunopulos (PAMI 2002). It computes, for each query x0,
    a weight vector w(x0) over features by comparing:
        (a) a full local class posterior P̂(j|z) around each local reference
            neighbor z of x0 (N1 neighborhood), and
        (b) the expected full local posterior bar{P̂}(j|x_i = z_i), obtained
            by averaging the full posterior over the remaining coordinates
            within a local slab around the i-th coordinate (N2 + Δ-slab).
    Relevance per feature is the Chi-squared distance between (a) and (b),
    aggregated over the K0 neighbors of the query; weights are an exponential
    transform of the (inverted) relevances.

- WeightedKNN
    Helper kNN that uses a per-feature weight vector inside the (weighted)
    Euclidean distance. Supports subspace sampling (with/without replacement)
    driven by a provided weight vector.

- KNNEClassifier
    A lightweight KNN-ensemble to demonstrate how ADAMENN weights can drive
    feature-subspace sampling and voting (simple / counting / Borda).
    It uses the WeightedKNN class as base classifier.

    
ADAMENN

Parameter notes:
----------------------------------------------------
• K0 : neighborhood of the query x0 used to *average* per-feature relevance
       (cf. Eq. (4) and Fig. 1 steps 2–4).
• K1 : neighborhood size for estimating the full local posterior P̂(j|z) (Eq. 7).
• K2 : neighborhood size for estimating the expected full local posterior
       bar_P̂(j|x_i = z_i) via a slab on coordinate i inside N2(z) (K2 > K1) (Eq. 8).
• L  : number of points in the Δ-slab on coordinate i (adaptive bandwidth).
• c  : positive factor in the exponential weighting scheme (Eq. 5).
• We iterate the compute–update loop `n_iter` times.

Implementation notes:
----------------------------------------------------
• J classes: C = {c_1, ..., c_J}, c_i ∈ [0, C-1].
• N training points
• q features: x = (x_1, ..., x_q) ∈ R^q

For one query point x0:
• Query point: x0 ∈ R^q => Weight vector: w(x0) = {w_1, ..., w_q} ∈ R^q (goal of ADAMENN) 
--> This expresses which features are relevant for predicting the class of x0
• K0 neighbors of x0: N(x0) = {z_1, ..., z_K0}; z_i ∈ R^q; each z_i is a local reference point of x0 
--> Use z_i ∈ N(x0) to compute local feature relevance of query point x0
• K1 neighbors of each z in N(x0): N1(z) = {u_1, ..., u_K1} with u_j ∈ R^q 
--> Use u_j ∈ N1(z) to compute full local class posterior P̂(j|z) of local reference point z using all of its features
• K2 neighbors of each z in N(x0): N2(z) = {v_1, ..., v_K2} with v_j ∈ R^q 
• L neighbors in the Δ-slab on feature i inside N2(z): S_i(z) = {s_1, ..., s_L}, s_j ∈ R^q; S_i(z) ∈ N2(z)
    - S_i(z) = {s ∈ N2(z) | s is adjacent to z if account for feature i only}
    - |S_i(z)| = L
--> Use S_i(z) to compute expected full local class posterior bar_P̂(j|x_i = z_i) of local reference point z on its feature i only

Steps: To attain a weight vector that represents the relevance of each feature i for a query point x0 (goal)
(1) Initialize weights w_i(x0) = 1 for all features i = 1, ..., q
(2) For a query point x0, find its K0 neighbors N(x0) = {z_1, ..., z_K0}
(3) Compute the relevance measure for feature i of x0 by:
        For each z in N(x0):
            Find its K1 neighbors N1(z) = {u_1, ..., u_K1} and compute full local class posterior P̂(j|z) of z (Eq. 7)
            Find its K2 neighbors N2(z) = {v_1, ..., v_K2}
                Find the L neighbors in the Δ-slab (based on feature i only) S_i(z) ⊆ N2(z) and compute expected full local class posterior bar_P̂(j|x_i = z_i) of z on its feature i only (Eq. 8)
            Compute the relevance discrepency r_i(z) between P̂(j|z) and bar_P̂(j|x_i = z_i) using Chi-squared distance (Eq. 3)
        Relevance measure bar_r_i(x0) for feature i is the average of r_i(z) over all z in N(x0) (Eq. 4)
(4) Repeat step (3) for all features i = 1, ..., q --> Get relevance measures: bar_R(x0) = {bar_r_1(x0), ..., bar_r_q(x0)}
(5) Acquire R(x0) = {R_i(x0) | R_i(x0) = max(bar_R(x0)) - bar_r_i(x0)}
(6) Compute the weight w(x0) = {w_1, ..., w_q} by applying an exponential weighting scheme on relevance measures R(x0) (Eq. 5)
(7) Repeat steps (2)–(6) for `n_iter` iterations to refine the weights

For a batch of m query points X0 = {x0_1, ..., x0_m}:
• Query matrix: X0 ∈ R^{m x q} => Weight matrix: W(X0) ∈ R^{m x q}
• K0 neighbors of each x0 in X0: N(X0) = {N(x0_1), ..., N(x0_m)} with N(x0_i) = {z_i1, ..., z_iK0} => Tensor: N(X0) ∈ R^{m x K0 x q}
• K1 neighbors of each z in N(X0): N1(N(X0)) = {N1(z_1_1), ..., N1(z_m_K0)} with N1(z_i_j) = {u_i_j1, ..., u_i_jK1} => Tensor: N1(N(X0)) ∈ R^{m x K0 x K1 x q}
• K2 neighbors of each z in N(X0): N2(N(X0)) = {N2(z_1_1), ..., N2(z_m_K0)} with N2(z_i_j) = {v_i_j1, ..., v_i_jK2} => Tensor: N2(N(X0)) ∈ R^{m x K0 x K2 x q}
• L neighbors in the Δ-slab on feature i inside N2(z) for each z in N(X0): S_i(N(X0)) = {S_i(z_1_1), ..., S_i(z_m_K0)} with S_i(z_i_j) = {s_i_j1, ..., s_i_jL} => Tensor: S_i(N(X0)) ∈ R^{m x K0 x L x q}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def pairwise_weighted_distances(
    X0: np.ndarray, 
    X: np.ndarray, 
    W: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise weighted Euclidean distances between two sets of points (X0 and X).

    Parameters
    ----------
    X0 : (m, q) array
        First set of points.
    X : (N, q) array
        Second set of points.
    W : (m, q) array
        Feature weights for each row in X0.

    Returns
    -------
    D : (m, N) array
        Pairwise weighted distances.
    """

    # Compute weighted squared differences and sum over features
    D = np.sqrt((W[:, np.newaxis, :] * (X0[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))                           # (m, N)                                   
    
    return D
    

# -----------------------------------------------------------------------------
# ADAMENN: local Chi-squared analysis -> exponential weights
# -----------------------------------------------------------------------------

@dataclass
class ADAMENNWeights:
    """
    ADAMENN feature relevance weights estimator.

    Parameters
    ----------
    K0 : int
        Neighborhood size for averaging per-feature relevance (Eq. 4).
    K1 : int
        Neighborhood size for estimating the full local posterior P̂(j|z) (Eq. 7).
    K2 : int
        Neighborhood size for estimating the expected full local posterior
        bar_P̂(j|x_i = z_i) via a slab on coordinate i inside N2(z) (K2 > K1) (Eq. 8).
    L : int
        Number of points in the Δ-slab on feature coordinate i (adaptive bandwidth) (Eq. 8).
    c : float
        Positive factor in the exponential weighting scheme (Eq. 5).
    n_iter : int
        Number of weight compute–update iterations.
    scale : bool, default=True
        Whether to standardize features to zero mean and unit variance.

    Attributes
    ----------
    X_std_ : (N, q) array
        Standardized training data (or just X if scale=False).
    y_ : (N,) array
        Integer class labels [0, C-1].
    classes_ : (C,) array
        Unique class labels (numerical: 0, ..., C-1).
    scaler_ : StandardScaler or None
        Fitted scaler when scale=True.
    """
    K0: int
    K1: int 
    K2: int
    L: int
    c: float
    n_iter: int = 1
    scale: bool = True

    # Fitted state
    X_std_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None
    classes_: Optional[np.ndarray] = None
    scaler_: Optional[StandardScaler] = None

    def __post_init__(self):
        if self.K0 <= 0 or self.K1 <= 0 or self.K2 <= 0 or self.c <= 0:
            raise ValueError("K0, K1, K2, and c must be positive.")
        if self.K2 <= self.K1:
            raise ValueError("K2 must be greater than K1.")
        if self.L > self.K2:
            raise ValueError("L must be less than or equal to K2.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> ADAMENNWeights:
        """
        Fit the ADAMENN weights estimator on training data.

        Parameters
        ----------
        X : (N, q) array
            Training data.
        y : (N,) array
            Class labels.

        Returns
        -------
        self : ADAMENNWeights
            Fitted estimator.
        """
        if self.K0 >= X.shape[0] or self.K1 >= X.shape[0] or self.K2 >= X.shape[0]:
            raise ValueError("K0, K1, and K2 must be less than the number of training samples.")
        
        self.classes_ = np.unique(y)
        self.y_ = y.copy()

        if self.scale:
            self.scaler_ = StandardScaler()
            X_std = self.scaler_.fit_transform(X)
        else:
            X_std = X.copy()
        self.X_std_ = X_std

        return self
    
    def _full_local_posterior_batch(
        self,
        N_X0: np.ndarray,
        K1: int
    ) -> np.ndarray:
        """
        Estimate local class posteriors P̂(j|z) for all local reference points z ∈ N(X0).

        Parameters
        ----------
        N_X0 : (m, K0, q) array
            All local reference points z of all m query points in the batch.
        K1 : int
            Number of neighbors defining the neighborhood N1(z).

        Returns
        -------
        post_all : (m, K0, C) array
            P̂(j|z) for every z ∈ N_X0, for each label in J.
        """
        X = self.X_std_
        y = self.y_
        N, q = X.shape
        C = len(self.classes_)
        
        # Collapse (m, K0, q) -> (m * K0, q)
        m, K0, _ = N_X0.shape
        Z = N_X0.reshape(-1, q)                                                                                                  # (m * K0, q)

        # Pairwise squared distances from each z to all X -> (m * K0, N)
        # Using squared distances (monotone w.r.t. Euclidean) avoids sqrt (faster)
        diff = Z[:, np.newaxis, :] - X[np.newaxis, :, :]                                                                         # (m * K0, N, q)
        d2 = np.einsum('mnq,mnq->mn', diff, diff)                                                                                # (m * K0, N)

        # Indices of K1 smallest distances per row
        K1_idx = np.argpartition(d2, K1 - 1, axis=1)[:, :K1]                                                                     # (m * K0, K1)
        # Gather neighbor labels
        y_N1 = y[K1_idx]                                                                                                         # (m * K0, K1)
        
        # Count labels per row into C bins, vectorized via one-hot + sum
        # One_hot: (n * K0, K1, C) -> sum over neighbors -> (m * K0, C)
        counts = np.eye(C, dtype=float)[y_N1].sum(axis=1)                                                                        # (m * K0, C)

        # Normalize by K1, then reshape back to (m, K0, C)
        post_Z = (counts / K1).reshape(m, K0, C)
        return post_Z

    def _expected_full_local_posterior_given_xi_batch(
        self,
        N_X0: np.ndarray,          
        K2: int,              
        L: int, 
    ) -> np.ndarray:
        """
        Estimate local class posteriors bar_P̂(j|x_i=z_i) for all z ∈ N_X0 and all features i

        Procedure:
        1) Build N2(z): the K2 nearest neighbors of z in FULL feature space.
        2) Within N2(z), for each feature i, select the L points whose coordinate i
            is closest to z_i (this defines the adaptive bandwidth Δ_i as the L-th smallest |x_{ni}-z_i|).
        3) Return the class histogram over those L points, normalized by L, in J's order.

        Parameters
        ----------
        N_X0 : (m, K0, q) array
            All local reference points z of all m query points in the batch.
        K2 : int
            Number of neighbors defining the neighborhood N2(z), must be > K1.
        L : int
            Number of points within the adaptive Δ interval to consider.

        Returns
        -------
        post_i_all : (m, K0, q, C)
            For each z (m, K0) and feature i (q), the posterior over classes (C).
        """
        X = self.X_std_
        y = self.y_
        N, q = X.shape
        C = len(self.classes_)

        # Flatten all z points: (m, K0, q) -> (m * K0, q)
        m, K0, _ = N_X0.shape
        Z = N_X0.reshape(m * K0, q)                                                                                              # (m * K0, q)

        # Pairwise squared distances from each z to all X -> (m * K0, N)
        # Using squared distances (monotone w.r.t. Euclidean) avoids sqrt (faster)
        diff = Z[:, None, :] - X[None, :, :]                                                                                     # (m * K0, N, q)
        d2 = np.einsum('mnq,mnq->mn', diff, diff)                                                                                # (m * K0, N)

        # Indices of K2 nearest neighbors for each z 
        K2_idx = np.argpartition(d2, K2 - 1, axis=1)[:, :K2]
        # Gather the neighbor points and labels
        X_N2 = X[K2_idx]                                                                                                         # (m * K0, K2, q)
        y_N2 = y[K2_idx]                                                                                                         # (m * K0, K2)

        # For each feature i, compute di = |x_{ni} - z_i| within N2(z) 
        di = np.abs(X_N2 - Z[:, None, :])                                                                                        # (m * K0, K2, q)

        # For each (z, i), select indices (within the K2 axis) of the L smallest distances.
        # argpartition along axis=1 (K2), then take first L
        slab_pos = np.argpartition(di, L - 1, axis=1)[:, :L, :]                                                                  # (m * K0, L, q)                                              

        # Gather the labels at those positions for each (z, i):
        # Expand y_N2 to (m * K0, K2, q) to align with slab_pos, then take along axis=1.
        y_K2_exp = np.repeat(y_N2[:, :, None], q, axis=2)                                                                        # (m * K0, K2, q)
        y_sel = np.take_along_axis(y_K2_exp, slab_pos, axis=1)                                                                   # (m * K0, L, q)

        # One-hot encode and sum over the L selected neighbors to get class counts:
        # one_hot -> (m * K0, L, q, C)  -> sum over axis=1 => (m * K0, q, C)
        counts = np.eye(C, dtype=float)[y_sel].sum(axis=1)                                                                       # (m * K0, q, C)

        # Normalize by L to get per-(z,i) posteriors, then reshape back to (m, K0, q, C)
        post_i_all = (counts / L).reshape(m, K0, q, C)                                                                           # (m, K0, q, C)
        return post_i_all

    def _weighted_chisq_distance_batch(
        self,
        post_Z: np.ndarray,
        post_Zi: np.ndarray,
        eps: float = 1e-12
    ):
        """
        Compute pairwise weighted chi-squared distance between the full local class posteriors and the expected full local posteriors given feature i.

        Parameters
        ----------
        post_Z : (m, K0, C) array
            The full local class posterior P̂(j|z) per z for all queries m.
        post_Zi : (m, K0, q, C) array
            The expected full local class posterior bar_P̂(j|x_i=z_i) per (z, i) for all queries m.
        eps : float, optional
            Small constant to avoid division by zero, by default 1e-12.

        Returns
        -------
        R : (m, K0, q)
            Weighted chi-squared distance per (z, i).
        """
        # Expand post_Z to broadcast across q: (m, K0, 1, C) vs (m, K0, q, C)
        post_Z_expanded = post_Z[..., None, :]                                                                                   # (m, K0, 1, C)

        # Weighted chi-square across classes
        num = (post_Z_expanded - post_Zi) ** 2                                                                                   # (m, K0, q, C)
        den = post_Zi + eps                                                                                                      # (m, K0, q, C)

        R = np.sum(num / den, axis=-1)                                                                                           # sum over C -> (m, K0, q)
        return R

    def _local_feature_relevance_batch(
        self,
        R: np.ndarray,   
        K0: int,         
    ) -> np.ndarray:  
        """
        Compute local feature relevance R̄ for each sample and feature by averaging
        R over the K0 neighbor points.

        Parameters
        ----------
        R : (m, K0, q) array
            R[m, k, i] ≥ 0 is the weighted chi-square distance for sample m,
            neighbor k, feature i.
        K0 : int
            Number of neighbors in N(x0).

        Returns
        -------
        R_bar : (m, q) array
            Local feature relevance per sample and feature: R̄[m, i] ≥ 0.
        """
        # Average over the neighbor axis (axis=1)
        R_bar = R.sum(axis=1) / float(K0)                                                                                        # equivalently: r.mean(axis=1) -> (m x q)
        return R_bar

    def _exponential_weights(
        self,
        R_bar: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exponential weights per sample w_i ∝ exp(c * R_i) with R_i = max(R̄) - R̄_i done row-wise for R_bar with shape (m, q)
        
        Parameters
        ----------
        R_bar : (m, q) array
            Each row holds local feature relevance scores for a query.

        Returns
        -------
        W : (m, q) array
            For each query, weights are > 0 and sum to 1 across features.
        """
        R = R_bar.max(axis=1, keepdims=True) - R_bar                                                                             # (m, q)

        cR = self.c * R                                  
        # Numerical stability: subtract the row-wise max before exp; doesn't change probabilities 
        cR = cR - cR.max(axis=1, keepdims=True)

        num = np.exp(cR)                                                                                                         # (m, q)
        den = num.sum(axis=1, keepdims=True)                                                                                     # (m, 1)
        W = num / den                                                                                                            # (m, q)
        return W
    
    def compute_weights(self, X0: np.ndarray) -> np.ndarray:
        """
        Compute ADAMENN feature weights for query points.

        Parameters
        ----------
        X0 : (m, q) array
            Query points.

        Returns
        -------
        W : (m, q) array
            Weight matrix for query points.
        """
        if self.X_std_ is None or self.y_ is None:
            raise RuntimeError("The estimator has not been fitted yet. Please call 'fit' first.")
        
        X = self.X_std_
        y = self.y_
        m, q = X0.shape
        X0_std = self.scaler_.transform(X0) if self.scale else X0.copy()

        # Step 1: Initialize weights W_j_i = 1 for all i features of j queries
        W = np.ones((m, q), dtype=float)

        for _ in range(self.n_iter):
            # Step 2: Compute N(X0) using current W (Eq. 6)
            D2 = pairwise_weighted_distances(X0_std, X, W)                                                                       # (m, N)
            N_X0_idx = np.argsort(D2, axis=1)[:, :self.K0]                                                                       # (m, K0)
            N_X0_X = X[N_X0_idx]                                                                                                 # (m, K0, q)            

            # Step 3: Compute relevance measures R̄(X0) = {r̄(x0) ∀ x0 ∈ X0}; r̄(x0) = {r̄_i(x0) | i ∈ [0, q-1]} (Eq. 7 - 8)                   
            # (a) Compute full local class posterior P̂(j|z), z ∈ N(X0)
            post_Z = self._full_local_posterior_batch(N_X0_X, self.K1)                                                           # (m, K0, C) 
            # (b) Compute full local class posterior bar_P̂(j|x_i=z_i)                                                                             
            post_Zi = self._expected_full_local_posterior_given_xi_batch(N_X0_X, self.K2, self.L)                                # (m, K0, q, C)
            # (c) Compute the weighted chi-squared distance between P̂(j|z) and bar_P̂(j|x_i=z_i)
            R = self._weighted_chisq_distance_batch(post_Z, post_Zi)
            # (d) Compute the local feature relevance measures
            R_bar = self._local_feature_relevance_batch(R, self.K0) 

            # Step 4: Update weights
            W = self._exponential_weights(R_bar)

        return W 
    

# -----------------------------------------------------------------------------
# Weighted kNN with per-feature weights + subspace sampling
# -----------------------------------------------------------------------------

@dataclass
class WeightedKNN:
    """
    KNN with a per-feature weight vector inside the Euclidean distance.

    Parameters
    ----------
    k : int
        Number of neighbors for a base voter to consider.
    scale : bool, default=True
        Standardize features before distance computations.
    """
    k: int
    scale: bool = True

    # Fitted state
    X_std_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None
    classes_: Optional[np.ndarray] = None
    scaler_: Optional[StandardScaler] = None

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> WeightedKNN:
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : (N, q) array
            Training data.
        y : (N,) array
            Class labels.

        Returns
        -------
        self : WeightedKNN
            Fitted classifier.
        """        
        self.classes_ = np.unique(y)
        self.y_ = y.copy()

        if self.scale:
            self.scaler_ = StandardScaler()
            X_std = self.scaler_.fit_transform(X)
        else:
            X_std = X.copy()
        self.X_std_ = X_std

        return self
    
    def _sample_feature_subpspace(
        self,
        W: np.ndarray,
        NoF: int,
        with_replacement: bool = False,
        rng_seed: Optional[int | np.random.RandomState] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a feature subspace for each query according to probabilities `W`.

        Parameters
        ----------
        W : (m, q) array
            Weight matrix for query points.
            W = {w_1, ..., w_m} for m queries; w_i = {w_i_1, ..., w_i_q} for q features.
        NoF : int
            Number of features for a voter to sample per query.
        with_replacement : bool, default=False
            If True, sample with replacement (the same feature can be drawn multiple times).
            If False, sample without replacement (each selected feature is unique).
        rng_seed : Optional[int | np.random.RandomState], default=None
            Seed or RandomState for reproducibility.
        
        Returns
        -------
        selected_idx : (m, NoF) array
            Indices drawn for each query (may contain duplicates when with_replacement=True).
        W_mask : (m, q) array
            Weight mask to use in the metric:
                - without replacement: W_mask[i, j] = W[i, j] for selected j, else 0.
                - with replacement:    W_mask[i, j] = multiplicity(j) * W[i, j], else 0.
        """
        m, q = W.shape
    
        if NoF <= 0:
            raise ValueError("NoF must be positive.")
        if NoF > q and not with_replacement:
            raise ValueError("If with_replacement=False, then NoF must be <= q.")

        rng = rng_seed if isinstance(rng_seed, np.random.RandomState) else check_random_state(rng_seed)

        selected_idx = np.empty((m, NoF), dtype=int)
        W_mask = np.zeros((m, q), dtype=float)

        for entry in range(m):
            p = W[entry]
            
            draws = rng.choice(q, size=NoF, replace=with_replacement, p=p)
            selected_idx[entry] = draws

            if with_replacement:
                counts = np.bincount(draws, minlength=q).astype(float)
                W_mask[entry] = counts * p
            else:
                W_mask[entry, draws] = p[draws]
        
        return selected_idx, W_mask
    
    def kneighbor_labels(
        self,
        X0: np.ndarray,
        W: np.ndarray,
        NoF: int,
        with_replacement: bool = False,
        rng_seed: Optional[int | np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Find the labels of the k nearest neighbors of X0 restricted to the selected features and
        using the per-feature weights computed.

        Parameters
        ----------
        X0 : (m, q) array
            Query points.
        W, NoF, with_replacement, rng_seed : (m, q) array, int, bool, Optional[int | np.random.RandomState]
            Method `_sample_feature_subpspace` hyperparameters.

        Returns
        ----------
        yk : (m, k) array
            k nearest neighbor labels in increasing distance.
        """

        if self.X_std_ is None or self.y_ is None:
            raise RuntimeError("The classifier has not been fitted yet. Please call 'fit' first.")
        
        X = self.X_std_
        y = self.y_
        X0_std = self.scaler_.transform(X0) if self.scale else X0.copy()
        _, W_mask = self._sample_feature_subpspace(W, NoF, with_replacement, rng_seed)                                           # (m, q)

        # Weighted squared distances
        D2 = pairwise_weighted_distances(X0_std, X, W_mask)                                                                      # (m, N)
        k_idx = np.argsort(D2, axis=1)[:, :self.k]                                                                               # (m, k)

        # Get neighbor labels for queries
        yk = y[k_idx]                                                                                                            # (m, k)

        return yk

        
# -----------------------------------------------------------------------------
# KNNE (weight-driven subspaces + voting)
# -----------------------------------------------------------------------------

@dataclass
class KNNEClassifier:
    """
    KNNE Classifier combining ADAMENN and the WeightedKNN ensemble.

    Parameters
    ----------
    K0, K1, K2, L, c, n_iter : ADAMENN hyperparameters
    k : int
        Number of neighbors for each base voter.
    NoF : int
        Number of features sampled per voter.
    with_replacement: bool
        Feature subspace sampling mode.
    NoC: int
        Number of voters in the ensemble.
    voting : str ('simple' | 'counting' | 'borda')
        Voting method to combine the decisions of member classifiers.
    scale : bool, default=True
        Whether to standardize features to zero mean and unit variance.
    """
    K0: int
    K1: int
    K2: int
    L: int
    c: float
    n_iter : int
    k: int
    NoF: int
    with_replacement: bool
    NoC: int
    voting: Literal['simple', 'counting', 'borda'] = 'borda'
    scale: bool = True     

    # fitted
    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None
    classes_: Optional[np.ndarray] = None
    
    adamenn_: Optional[ADAMENNWeights] = None
    knns_: Optional[List[WeightedKNN]] = None
    rng_: Optional[int | np.random.RandomState] = None

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray, 
        rng_seed: Optional[int | np.random.RandomState] = None
    ) -> KNNEClassifier:
        """
        Fit the classifier ensemble on training data.

        Parameters
        ----------
        X : (N, q) array
            Training data.
        y : (N,) array
            Class labels.
        rng_seed : Optional[int | np.random.RandomState]
            Seed or RandomState for reproducibility.

        Returns
        -------
        self : sEClassifier
            Fitted classifier ensemble.
        """        
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        self.adamenn_ = ADAMENNWeights(
            K0=self.K0, K1=self.K1, K2=self.K2, L=self.L, c=self.c,
            n_iter=self.n_iter, scale=self.scale
        ).fit(X, y)

        self.knns_ = [WeightedKNN(k=self.k, scale=self.scale).fit(X, y) for i in range(self.NoC)]

        self.rng_ = rng_seed if isinstance(rng_seed, np.random.RandomState) else check_random_state(rng_seed)

        return self
    
    def _ensemble_knn_labels(
        self,
        X0: np.ndarray,
    ) -> np.ndarray:
        """
        For each voter in the ensemble of size NoC, get its (m, k) neighbor labels,
        stacked to (m, NoC, k).

        Parameters
        ----------
        X0 : (m, q) array
            Query points.

        Returns
        -------
        knn_labels : (m, NoC, k) array
            Labels of k neighbors / classifier, for NoC classifiers / query, for m queries.
        """
        m = X0.shape[0]
        NoC = self.NoC
        k = self.k

        # ADAMENN per-query weights
        W = self.adamenn_.compute_weights(X0)                                                                                    # (m, q)
        # container for all voters' neighbor labels
        knn_labels = np.empty((m, NoC, k), dtype=int)

        # One RNG to diversify voters (each voter gets its own subspace draw)
        INT32_MAX_EXCL = np.iinfo(np.int32).max 
        rng = self.rng_ if isinstance(self.rng_, np.random.RandomState) else check_random_state(self.rng_)

        for c, knn in enumerate(self.knns_):
            seed_c = int(rng.randint(0, INT32_MAX_EXCL))
            yk = knn.kneighbor_labels(
                X0, W, self.NoF,
                with_replacement=self.with_replacement,
                rng_seed=seed_c
            )                                                                                                                    # (m, k)
            knn_labels[:, c, :] = yk

        return knn_labels                                                                                                        # (m, NoC, k)
    
    def predict(
        self, 
        X0: np.ndarray, 
    ) -> np.ndarray:
        """
        Predict class labels for queries.

        Steps:
            1) Compute ADAMENN weights.
            2) Run an ensemble of WeightedKNN voters in subspaces sampled 
            from the weight distributions.
            3) Vote final labels using the chosen strategy.

        Parameters
        ----------
        X0 : (m, q) array
            Query points.

        Returns
        -------
        y_pred : (m,) int array
        """
        if self.adamenn_ is None or self.knns_ is None:
            raise RuntimeError("The classifier ensemble has not been fitted yet. Please call 'fit' first.")
        
        m = X0.shape[0]
        knn_labels = self._ensemble_knn_labels(X0)                                                                               # (m, NoC, k)

        # Voting
        C = len(self.classes_)
        y_pred = np.empty(m, dtype=int)

        if self.voting == 'simple':
            # Base voter prediction (Majority among its k neighbors)
            base_preds = np.apply_along_axis(
                lambda row: np.bincount(row, minlength=C).argmax(),
                axis=2,
                arr=knn_labels
            )                                                                                                                    # (m, NoC)
            # Final majority vote across all NoC
            for i in range(m):
                counts = np.bincount(base_preds[i], minlength=C)
                y_pred[i] = counts.argmax()

        elif self.voting == 'counting':
            # Pool all k neighbors from all NoC voters (m, NoC * k) and take majority
            pooled = knn_labels.reshape(m, -1)
            for i in range(m):
                counts = np.bincount(pooled[i], minlength=C)
                y_pred[i] = counts.argmax()

        elif self.voting == 'borda':
            # knn_labels: (m, NoC, k) with labels 0, ..., C-1
            m, NoC, k = knn_labels.shape

            # 1) Class counts per voter: (m, NoC, C)
            one_hot = np.eye(C, dtype=int)[knn_labels]                                                                          # (m, NoC, k, C)
            voter_counts = one_hot.sum(axis=2)                                                                                  # (m, NoC, C)

            # 2) Rank classes by count (ascending) per (m, voter), stable for tie determinism
            ranks_idx = np.argsort(voter_counts, axis=-1, kind='mergesort')                                                     # (m, NoC, C)

            # 3) Convert order indices -> positions (0, ..., C-1), where C-1 is best (highest count). 
            # This is the Borda score per voter.
            score_ranks = np.empty_like(ranks_idx)
            ar_m = np.arange(m)[:, None, None]
            ar_v = np.arange(NoC)[None, :, None]
            ar_pos = np.arange(C)[None, None, :]
            score_ranks[ar_m, ar_v, ranks_idx] = ar_pos                                                                         # (m, NoC, C); higher is better

            # 4) Sum Borda points across voters and pick argmax per query
            total_borda_score = score_ranks.sum(axis=1)                                                                         # (m, C)
            y_pred = total_borda_score.argmax(axis=1)                                                                           # (m,)

        else:
            raise ValueError(f"Unknown voting method: {self.voting}")

        return y_pred

                                                                                           
        


