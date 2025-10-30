"""
mibos.py

An implementation of the MIBOS methhod, a similarity-based iterative approach 
to categorical data imputation.

This module defines the MIBOS class, which provides methods to fit the imputer
on an incomplete dataset and perform imputation using the MIBOS algorithm.

References
----------
• Zhang, S., Chen, W., & Li, Y. (2021). MIBOS: A similarity-based iterative approach to categorical data imputation.
  IEEE Transactions on Knowledge and Data Engineering, 34(3), 1124-113
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Literal, Union

import numpy as np
import pandas as pd

@dataclass
class MIBOS:
    """
    MIBOS imputer for categorical data.
    
    Parameters
    ----------
    missing_token : str, default='nan'
        The token representing missing values in the dataset. 
        • If 'nan': Standard missing NaNs (np.nan / None / pd.NA).
        • Else: Custom string token representing missing values.
    order_strategy : {"none", "ascending", "decsending", "DA"}, default="none"
        Strategy to determine the sample row order of imputation.
        • "none": Single imputing direction (in input row order). This is the basic MIBOS algorithm.
        • "acsending": Impute samples from least to most missing attributes.
        • "descending": Impute samples from most to least missing attributes.
        • "DA": Perform both directions of "ascending" and "descending", then merge the results (prefer "descending" where conflicts).
                --> Descending result plus any extra cells the ascending pass was able to fill where descending could not.
                This is the extended MIBOS algorithm: MIBOS-DA.
    fallback : {"mode", "uniform"}, default="mode"
        Fallback strategy to fill any leftovers after convergence using per-column mode.
        • "mode": Impute with the most frequent category of the attribute in the training data.
        • "uniform": Impute with a random category sampled uniformly from the attribute's categories in the training data.

    Attributes
    ----------
    X_ : (N, c) array
        The incomplete dataset used for imputation (only contains categorical features, missing).
        This set excludes the class label.
    """
    # Attributes
    missing_token: str = 'nan'
    order_strategy: Literal["none", "ascending", "descending", "DA"] = "none"
    fallback: Literal["mode", "uniform"] = "mode"

    # Fitted state
    X_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> MIBOS:
        """
        Fit the MIBOS imputer on the incomplete dataset.
        
        Parameters
        ----------
        X : (N, c) dataframe or array
            The incomplete dataset.

        Returns
        -------
        self : MIBOS
            Fitted MIBOS imputer.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.astype(object).to_numpy(copy=True)
        else:
            arr = np.asarray(X, dtype=object).copy()

        # Standard missing (np.nan / None / pd.NA)
        na_mask = pd.isna(arr)

        if self.missing_token == 'nan':
            # Replace standard missing with the string 'np.nan'
            if na_mask.any():
                arr[na_mask] = self.missing_token
        else:
            # Also treat the literal token as missing
            tok_mask = (arr == self.missing_token)
            miss_mask = na_mask | tok_mask
            if miss_mask.any():
                arr[miss_mask] = self.missing_token

        self.X_ = np.ascontiguousarray(arr, dtype=object)
        return self
        
    def _compute_object_similarity(self, 
        xi: np.ndarray, 
        xj: np.ndarray
    ) -> int:
        """
        Compute the similarity between two samples based on the number of matching non-missing categorical attributes.
        Return 0 if any jointly-observed attribute differs; else count matches.
        
        Parameters
        ----------
        xi : (c,) array
            First sample.
        xj : (c,) array
            Second sample.
        
        Returns
        -------
        sim : int
            Similarity score between the two samples.
        """
        obs_i = xi != self.missing_token
        obs_j = xj != self.missing_token
        both  = obs_i & obs_j

        # If no jointly observed attributes, similarity = 0
        if not np.any(both):
            return 0
        # If any jointly observed attributes differ, similarity = 0
        if np.any(xi[both] != xj[both]):
            return 0
        
        # No conflicts: similarity = count of matches among observed
        return int(np.sum(xi[both] == xj[both]))
    
    def _compute_similarity_vector(
        self,
        xi: np.ndarray,
        X_current: np.ndarray
    ) -> np.ndarray:
        """ 
        Compute the similarity vector of a missing sample xi, where each entry represents the object similarity between
        this sample with every other sample in the dataset.

        Parameters
        ----------
        xi : (c,) array
            The missing sample.
        Returns
        -------
        m : (N,) array
            Similarity vector. 
        """
        X = X_current.copy()
        N = X.shape[0]

        m = np.zeros((N,), dtype=int)
        for j in range(N):
            if X[j] is xi:
                continue
            m[j] = self._compute_object_similarity(xi, X[j])

        return m
    
    def _missing_attribute_set(
        self,
        xi: np.ndarray,
    ) -> np.ndarray:
        """
        Identify the set of missing attribute indices for the missing sample xi.

        Parameters
        ----------
        xi : (c,) array
            The missing sample.
        
        Returns
        -------
        miss_set : (k,) array
            Indices of missing attributes in the sample xi.
        """
        miss_set = np.where(xi == self.missing_token)[0]
        return np.array(miss_set, dtype=int
    )

    def _nearest_undifferentiated_set(
        self,
        xi: np.ndarray,
        sim_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Identify the nearest undifferentiated set for the missing sample xi based on the similarity vector.

        Parameters
        ----------
        xi : (c,) array
            The missing sample.
        sim_vec : (N,) array
            Similarity vector of the missing sample.
        
        Returns
        -------
        undiff_set : (k,) array
            Indices of the nearest undifferentiated set in the dataset X.
        """
        max_sim = np.max(sim_vec)
        if max_sim == 0:
            return np.array([], dtype=int)

        undiff_set = np.where(sim_vec == max_sim)[0]

        return np.array(undiff_set, dtype=int)
    
    def _compute_sample_order(
        self,
        X_current: np.ndarray,
        strategy: str,
    ) -> np.ndarray:
        """
        Compute the sample row order for imputation based on the specified order strategy.

        Parameters
        ----------
        X_current : (N, c) array
            The current incomplete dataset.
        strategy : str
            The order strategy to use ("none", "ascending", "descending").

        Returns
        -------
        order : array
            List of sample indices representing the imputation order.
        """
        X = X_current.copy()
        missing_counts = np.sum(X == self.missing_token, axis=1)

        if strategy == "none":
            order = np.where(np.any(X == self.missing_token, axis=1))[0]   
        elif strategy == "ascending":
            order = np.argsort(missing_counts)
        elif strategy == "descending":
            order = np.argsort(-missing_counts)
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of 'none', 'ascending', 'descending'.")

        return order

    def _perform_ordered_imputation(
        self,
        strategy: str,
    ) -> np.ndarray:
        """
        Perform MIBOS imputation on the fitted incomplete dataset.
        
        Parameters
        ----------
        strategy : str
            The order strategy to use ("none", "ascending", "descending").

        Returns
        -------
        X_imputed : (N, c) array
            The imputed dataset.
        """
        if self.X_ is None:
            raise RuntimeError("MIBOS imputer has not been fitted yet. Please call 'fit' with the incomplete dataset before calling 'impute'.")
        
        X_imputed = self.X_.copy()
            
        while True:
            missing_idx = self._compute_sample_order(X_imputed, strategy)            

            num = 0
            for i in missing_idx:
                xi = X_imputed[i]
                sim_vec = self._compute_similarity_vector(xi, X_imputed)
                
                MAS_i = self._missing_attribute_set(xi)
                if MAS_i.size == 0:
                    continue

                NS_i = self._nearest_undifferentiated_set(xi, sim_vec)
                if NS_i.size == 0:
                    continue
                elif NS_i.size == 1:
                    xj = X_imputed[NS_i[0]]
                    for attr in MAS_i:
                        if xj[attr] != self.missing_token:
                            xi[attr] = xj[attr]
                            num += 1
                else:
                    for attr in MAS_i:
                        vals = X_imputed[NS_i, attr]
                        mask = (vals != self.missing_token)
                        observed_vals = vals[mask]
                        if observed_vals.size == 0:
                            continue
                        unique_values = np.unique(observed_vals)
                        if unique_values.size == 1:
                            xi[attr] = unique_values[0]
                            num += 1
            if num == 0:
                break
        
        return X_imputed
    
    def _merge_imputations(
        self,
        X_desc: np.ndarray,
        X_asc: np.ndarray
    ) -> np.ndarray:
        """
        Merge the results of descending and ascending imputations for MIBOS-DA.

        Parameters
        ----------
        X_desc : (N, c) array
            The imputed dataset from descending order.
        X_asc : (N, c) array
            The imputed dataset from ascending order.

        Returns
        -------
        X_merged : (N, c) array
            The merged imputed dataset.
        """
        X_merged = X_desc.copy()
        N, c = X_merged.shape

        for i in range(N):
            for j in range(c):
                if X_merged[i, j] == self.missing_token and X_asc[i, j] != self.missing_token:
                    X_merged[i, j] = X_asc[i, j]

        return X_merged
    
    def _fallback_imputation(
        self,
        X_imputed: np.ndarray
    ) -> np.ndarray:
        """
        Perform fallback imputation on any remaining missing values in the dataset.

        Parameters
        ----------
        X_imputed : (N, c) array
            The imputed dataset with possible remaining missing values.

        Returns
        -------
        X_fallback : (N, c) array
            The dataset after fallback imputation.
        """
        X_fallback = X_imputed.copy()
        N, c = X_fallback.shape

        for j in range(c):
            col = X_fallback[:, j]
            missing_mask = (col == self.missing_token)
            if not np.any(missing_mask):
                continue

            observed_vals = col[~missing_mask]
            if observed_vals.size == 0:
                continue

            if self.fallback == "mode":
                values, counts = np.unique(observed_vals, return_counts=True)
                mode_val = values[np.argmax(counts)]
                col[missing_mask] = mode_val
            elif self.fallback == "uniform":
                unique_vals = np.unique(observed_vals)
                random_choices = np.random.choice(unique_vals, size=np.sum(missing_mask), replace=True)
                col[missing_mask] = random_choices
            else:
                raise ValueError(f"Invalid fallback strategy: {self.fallback}. Must be one of 'mode', 'uniform'.")

            X_fallback[:, j] = col

        return X_fallback
    
    def impute(self) -> np.ndarray:
        """
        Perform complete MIBOS imputation procedure on the fitted incomplete dataset with fallback imputation.

        Returns
        -------
        X_final : (N, c) array
            The fully imputed dataset.
        """
        if self.order_strategy in ["none", "ascending", "descending"]:
            X_imputed = self._perform_ordered_imputation(strategy=self.order_strategy)
        elif self.order_strategy == "DA":
            X_imputed_desc = self._perform_ordered_imputation(strategy="descending")
            X_imputed_asc = self._perform_ordered_imputation(strategy="ascending")
            X_imputed = self._merge_imputations(X_imputed_desc, X_imputed_asc)
        else:
            raise ValueError(f"Invalid order_strategy: {self.order_strategy}. Must be one of 'none', 'ascending', 'descending', 'DA'.")

        if not np.any(X_imputed == self.missing_token):
            return X_imputed

        X_final = self._fallback_imputation(X_imputed)
        return X_final


        
        


    



