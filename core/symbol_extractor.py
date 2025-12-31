import numpy as np

class SymbolExtractor:
    """
    Converts continuous latent vectors into discrete symbolic IDs
    via centroid assignment (lightweight k-means style).
    """

    def __init__(self, k=12):
        self.k = k
        self.centroids = None
        self.symbol_ids = [f"S{i}" for i in range(k)]

    def fit(self, latents):
        latents = np.array(latents)
        if len(latents) < self.k:
            return
        # simple deterministic centroid seeding
        idxs = np.linspace(0, len(latents) - 1, self.k).astype(int)
        self.centroids = latents[idxs]

    def assign(self, latent):
        if self.centroids is None:
            return "S0"
        latent = np.array(latent)
        dists = np.linalg.norm(self.centroids - latent, axis=1)
        return self.symbol_ids[int(np.argmin(dists))]
