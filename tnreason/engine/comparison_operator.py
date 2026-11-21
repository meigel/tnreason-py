"""Numerical comparison as an operator, not a matrix.

A comparison factor over entities, M[z,y] = 1[p_z < q_y], is dense by nature and must never be built.
It is applied instead as a cumulative sum: sort the left entities by their feature, take running sums of
the weight column, and read off the value at the count of entities below each threshold. The cost is
dominated by the sort, against n^2 entries for the materialised factor.

This is the construction used for numerical rules in the differentiable rule-learning literature, here
as an operator inside a contraction: the factor participates in a body exactly like a relational atom,
but it is applied rather than stored.
"""

import numpy as np
from scipy import sparse


class ComparisonOperator:
    """M[z, y] = 1[p[z] OP q[y]] for a bracket operator, applied by cumulative sums.

    Only the two-sided order is needed for `<` and `>`; equality and inequality fall back to a direct
    construction because they are not monotone.
    """

    def __init__(self, leftValues, rightValues, operator="<"):
        self.left = np.asarray(leftValues, dtype=np.float64)
        self.right = np.asarray(rightValues, dtype=np.float64)
        self.operator = operator
        if operator not in ("<", ">", "<=", ">="):
            raise NotImplementedError("operator {} is not monotone".format(operator))

    def __len__(self):
        return self.left.size

    def apply(self, weightMatrix, leftIsRows=True):
        """A @ M for a sparse A whose columns run over the left entities of the comparison."""
        weights = weightMatrix.tocsc() if sparse.issparse(weightMatrix) else sparse.csc_matrix(weightMatrix)
        if weights.shape[1] != self.left.size:
            raise ValueError("weight columns {} against {} left values".format(
                weights.shape[1], self.left.size))

        left = self.left
        right = self.right
        if self.operator in (">", ">="):
            left = -left
            right = -right
            strict = self.operator == ">"
        else:
            strict = self.operator == "<"

        order = np.argsort(left, kind="stable")
        sortedLeft = left[order]
        # how many left entities satisfy the comparison for each right entity
        counts = np.searchsorted(sortedLeft, right, side="left" if strict else "right")
        safeCounts = np.maximum(counts - 1, 0)

        result = np.zeros((weights.shape[0], right.size), dtype=np.float64)
        for row in range(weights.shape[0]):
            rowWeights = np.asarray(weights.getrow(row).todense()).ravel()[order]
            running = np.cumsum(rowWeights)
            result[row] = np.where(counts > 0, running[safeCounts], 0.0)
        return sparse.csr_matrix(result)


def materialized(leftValues, rightValues, operator="<"):
    """The dense factor, for tests only."""
    left = np.asarray(leftValues, dtype=np.float64)[:, None]
    right = np.asarray(rightValues, dtype=np.float64)[None, :]
    return {
        "<": left < right, ">": left > right,
        "<=": left <= right, ">=": left >= right,
    }[operator].astype(np.float64)


def _self_check():
    rng = np.random.default_rng(0)
    n = 25
    left = rng.integers(0, 40, size=n).astype(np.float64)
    right = rng.integers(0, 40, size=n).astype(np.float64)
    weights = sparse.random(6, n, density=0.4, random_state=1, format="csr")

    for operator in ("<", ">", "<=", ">="):
        fast = ComparisonOperator(left, right, operator).apply(weights)
        dense = weights.toarray() @ materialized(left, right, operator)
        difference = np.abs(fast.toarray() - dense).max()
        assert difference < 1e-10, "{} disagrees: {}".format(operator, difference)
        print("operator {}: matches the materialised factor, max difference {:.1e}".format(
            operator, difference))

    # an affine transform of one side is still a monotone comparison, so it stays a sort
    for alpha, beta in ((2.0, 3.0), (0.5, -1.0)):
        for operator in ("<", ">"):
            scaled = ComparisonOperator(alpha * left + beta, right, operator).apply(weights)
            dense = weights.toarray() @ materialized(alpha * left + beta, right, operator)
            difference = np.abs(scaled.toarray() - dense).max()
            assert difference < 1e-10, "affine {} {} disagrees: {}".format(alpha, operator, difference)
            print("affine alpha={} beta={} operator {}: matches to {:.1e}".format(
                alpha, beta, operator, difference))

    # a genuinely two-sided comparison is not a sort on either side: it needs a 2-D structure
    a2 = rng.random((n, 2))
    orderByFirst = np.argsort(a2[:, 0])
    blocks = np.searchsorted(a2[orderByFirst, 0], a2[:, 0])
    violation = 0
    for i in range(n):
        for j in range(n):
            if a2[i].sum() < a2[j].sum() and orderByFirst[blocks[i]] > orderByFirst[blocks[j]]:
                violation += 1
    print("two-sided sums: {} pairs where a 1-D sort on one column gives the wrong order".format(
        violation))

    full = 40 * 40
    stored = weights.nnz
    print("materialising the factor would store {} entries per body; applying it stores {}".format(
        full, stored))


if __name__ == "__main__":
    _self_check()
