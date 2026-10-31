"""Linear-algebra movie recommender pipeline."""
import numpy as np


def preprocess(R):
    """Center observed ratings by user mean; represent missing centered values as zero."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or min(R.shape) < 2:
        raise ValueError("Ratings must be a 2D matrix with at least 2 users and 2 movies.")
    mask = np.isfinite(R)
    if not mask.any(axis=1).all():
        raise ValueError("Every user must have at least one observed rating.")
    mu = np.array([row[observed].mean() for row, observed in zip(R, mask)])
    C = np.where(mask, R - mu[:, None], 0.0)
    return C, mask, mu


def rank_analysis(R, C):
    zero_filled = np.where(np.isfinite(R), R, 0)
    return {
        "rank_raw": int(np.linalg.matrix_rank(zero_filled)),
        "rank_centered": int(np.linalg.matrix_rank(C)),
        "singular_values": np.linalg.svd(C, compute_uv=False),
    }


def cosine_matrix(X):
    """Cosine similarity between rows, safely handling all-zero vectors."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1)
    denom = np.outer(norms, norms)
    return np.divide(X @ X.T, denom, out=np.zeros((X.shape[0], X.shape[0])), where=denom > 0)


def pca(C, k=2):
    """Eigen-decomposition of the movie covariance matrix."""
    if C.shape[0] < 2:
        raise ValueError("PCA requires at least two users.")
    cov = C.T @ C / (C.shape[0] - 1)
    eigenvalues, components = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, components = eigenvalues[order], components[:, order]
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return {"eigenvalues": eigenvalues, "explained": explained, "components": components, "k": k}


def project(C, V, k):
    return C @ V[:, :k]


def reconstruct(Z, V, mu, k):
    return np.clip(Z @ V[:, :k].T + mu[:, None], 1, 5)


def recommend(R, k=2):
    R = np.asarray(R, dtype=float)
    C, mask, mu = preprocess(R)
    if np.isinf(R).any():
        raise ValueError("Ratings matrix cannot contain infinite values.")
    p = pca(C, k)
    Z = project(C, p["components"], k)
    Rhat = reconstruct(Z, p["components"], mu, k)
    recs = []
    for i in range(R.shape[0]):
        for j in np.where(~mask[i])[0]:
            recs.append({
                "user": i,
                "movie": int(j),
                "predicted": float(Rhat[i, j]),
                "user_mean": float(mu[i]),
                "recommend": bool(Rhat[i, j] >= mu[i]),
            })
    return {
        "C": C, "mask": mask, "mu": mu,
        "rank": rank_analysis(R, C),
        "user_sim": cosine_matrix(C),
        "movie_sim": cosine_matrix(C.T),
        "pca": p, "Z": Z, "Rhat": Rhat, "recs": recs,
    }
