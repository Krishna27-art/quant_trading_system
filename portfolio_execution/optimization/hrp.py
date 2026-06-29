import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) Portfolio Optimization.
    Reference: 'Hierarchical Risk Parity and Minimum Variance Portfolio Design', Marcos Lopez de Prado
    """

    def __init__(self):
        pass

    def _get_distance_matrix(self, corr: pd.DataFrame) -> pd.DataFrame:
        """Calculate the distance matrix from the correlation matrix."""
        # Distance metric: d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))
        # Ensure correlations are strictly between -1 and 1
        corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(0.5 * (1 - corr))
        return dist

    def _get_quasi_diag(self, link: np.ndarray) -> list[int]:
        """
        Sort clustered items by distance.
        Recursively extract the sequence of leaves from a linkage matrix.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]  # number of original items

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _get_cluster_var(self, cov: pd.DataFrame, c_items: list[int]) -> float:
        """Calculate cluster variance."""
        cov_ = cov.iloc[c_items, c_items]  # matrix slice

        # Inverse variance portfolio weights
        ivp = 1.0 / np.diag(cov_)
        ivp /= ivp.sum()

        # Portfolio variance
        w = ivp.reshape(-1, 1)
        c_var = np.dot(np.dot(w.T, cov_), w)[0, 0]
        return c_var

    def _get_rec_bipart(self, cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
        """Recursive bisection allocation."""
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]  # initialize all items in one cluster

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i + 1]  # cluster 2

                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)

                alpha = 1 - c_var0 / (c_var0 + c_var1)

                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1 - alpha  # weight 2

        return w

    def optimize(self, returns: pd.DataFrame) -> dict[str, float]:
        """
        Calculates the HRP portfolio weights.

        :param returns: DataFrame of asset returns
        :return: Dictionary of {asset_name: weight}
        """
        cov = returns.cov()
        corr = returns.corr()

        # 1. Distance matrix
        dist = self._get_distance_matrix(corr)

        # 2. Linkage (Hierarchical Clustering)
        # Using the condensed distance matrix for linkage
        from scipy.spatial.distance import squareform

        # Ensure symmetric 0 diagonal
        np.fill_diagonal(dist.values, 0)
        condensed_dist = squareform(dist.values)
        link = sch.linkage(condensed_dist, method="single")

        # 3. Quasi-Diagonalization
        sort_ix = self._get_quasi_diag(link)

        # 4. Recursive Bisection
        weights = self._get_rec_bipart(cov, sort_ix)

        # Re-index to original asset names
        weights.index = returns.columns[sort_ix]
        weights = weights.sort_index()

        return weights.to_dict()
