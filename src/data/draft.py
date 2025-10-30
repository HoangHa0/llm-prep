def _compute_similarity_matrix(
    self
) -> np.ndarray:
    """ 
    Compute the similarity matrix, where each entry (i, j) represents the object similarity between
    sample xi and sample xj .

    Returns
    -------
    M : (N, N) array
        Similarity matrix. 
    """
    X = self.X_.copy()
    N = X.shape[0]
    M = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            M[i, j] = self._compute_object_similarity(X[i], X[j])

    return M