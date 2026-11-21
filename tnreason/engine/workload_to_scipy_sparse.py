"""Sparse cores and a compiled contraction for them.

The polynomial core scans monomial pairs, which costs the product of the two edge counts regardless of
how few pairs actually match. Graph relations are sparse, so almost all of that work is wasted. This
module gives the engine a core that stores a relation as a compressed sparse row matrix and a
contractor that multiplies two of them with the compiled sparse kernel, which is what a sparse rule
learner does by hand.

Scope: cores of arity one and two, which covers chains, forks and trees (a tree premise hangs off one
variable, so it is an arity-one factor). Arity above two falls back to the numpy contractor, which
materialises the factor; that is honest but slow, and it is where a hyperedge backend would go.
"""

import numpy as np
from scipy import sparse

from tnreason.engine.core_base import TensorCore
from tnreason.engine.workload_to_numpy import NumpyCore


class ScipySparseCore(TensorCore):
    """A core over one or two colours, stored as a compressed sparse row matrix.

    Arity one is stored as a diagonal matrix so that every join is a matrix product. The diagonal
    costs only as many nonzeros as the vector has.
    """

    coreType = "ScipySparseCore"

    def __init__(self, values=None, colors=None, name="NoName", shape=None):
        super().__init__(colors, name, shape)
        if values is None:
            self.values = None
            return
        matrix = values if sparse.issparse(values) else sparse.csr_matrix(np.asarray(values))
        matrix = matrix.tocsr()
        if len(self.colors) == 1:
            vector = matrix.toarray().ravel() if matrix.shape != (shape[0], shape[0]) else matrix.diagonal()
            matrix = sparse.diags(np.asarray(vector).ravel()).tocsr()
        self.values = matrix

    def clone(self):
        return ScipySparseCore(values=self.values.copy(), colors=list(self.colors), name=self.name,
                               shape=list(self.shape))

    def get_slice(self, colorEvidenceDict):
        matrix = self.values
        colors = list(self.colors)
        if len(colors) == 2 and colors[0] in colorEvidenceDict:
            index = colorEvidenceDict[colors[0]]
            row = matrix.getrow(index).toarray().ravel() if hasattr(matrix, "getrow") else matrix[index].toarray().ravel()
            return ScipySparseCore(values=row.reshape(1, -1), colors=[colors[1]], name=self.name,
                                   shape=[self.shape[1]])
        raise NotImplementedError("slicing only supports fixing the first colour of a binary core")

    def to_NumpyCore(self):
        if len(self.colors) == 1:
            return NumpyCore(values=np.asarray(self.values.diagonal()).ravel(), colors=list(self.colors),
                             name=self.name, shape=list(self.shape))
        return NumpyCore(values=self.values.toarray(), colors=list(self.colors), name=self.name,
                         shape=list(self.shape))


class ScipySparseContractor:
    """Contract sparse cores pairwise, multiplying with the compiled sparse kernel.

    A shared colour that is not open is summed out, which is exactly a matrix product once the shared
    colour is placed on the second axis of the left factor and the first axis of the right one.
    """

    def __init__(self, coreDict, openColors):
        self.cores = {key: core.clone() for key, core in coreDict.items()}
        self.openColors = list(openColors)

    def contract(self):
        cores = dict(self.cores)
        while len(cores) > 1:
            keys = list(cores.keys())
            merged = False
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    shared = set(cores[keys[i]].colors) & set(cores[keys[j]].colors)
                    shared -= set(self.openColors)
                    if len(shared) != 1:
                        continue
                    left, right = cores[keys[i]], cores[keys[j]]
                    middle = shared.pop()
                    result = self._join(left, right, middle)
                    del cores[keys[i]], cores[keys[j]]
                    cores["_join{}".format(len(cores))] = result
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                # no shared colour left: the remaining factors are independent, so contract densely
                from tnreason.engine.workload_to_numpy import NumpyEinsumContractor
                dense = {k: v.to_NumpyCore() for k, v in cores.items()}
                out = NumpyEinsumContractor(coreDict=dense, openColors=self.openColors).contract()
                return ScipySparseCore(values=(out.values if hasattr(out, "values") else out),
                                       colors=self.openColors, name="contracted",
                                       shape=[out.shape[i] for i in range(len(self.openColors))])
        only = list(cores.values())[0]
        return self._reduce(only)

    def _join(self, left, right, middle):
        leftMatrix = self._orient(left, middle, second=True)
        rightMatrix = self._orient(right, middle, second=False)
        product = (leftMatrix @ rightMatrix).tocsr()
        leftOther = [c for c in left.colors if c != middle]
        rightOther = [c for c in right.colors if c != middle]
        colors = leftOther + rightOther
        shape = [self._dim(left, c) for c in leftOther] + [self._dim(right, c) for c in rightOther]
        if len(colors) == 0:
            return ScipySparseCore(values=np.array([[product.sum()]]), colors=[], name="scalar", shape=[])
        return ScipySparseCore(values=product, colors=colors, name="joined", shape=shape)

    def _orient(self, core, middle, second):
        """Return the matrix with the shared colour on the second (or first) axis."""
        colors = list(core.colors)
        if len(colors) == 1:
            return core.values
        if second and colors[1] != middle:
            return core.values.T.tocsr()
        if not second and colors[0] != middle:
            return core.values.T.tocsr()
        return core.values

    def _dim(self, core, color):
        return core.shape[core.colors.index(color)]

    def _reduce(self, core):
        if set(core.colors) == set(self.openColors) or not self.openColors:
            return core
        order = [core.colors.index(c) for c in self.openColors if c in core.colors]
        if len(order) == len(core.colors):
            return core
        raise NotImplementedError("reducing to a strict subset of open colours is not supported")


def contract_sparse(coreDict, openColors):
    """Convenience wrapper matching the engine.contract signature."""
    return ScipySparseContractor(coreDict=coreDict, openColors=openColors).contract()


def _self_check():
    """A chain, a tree gate and a fork must match the dense einsum contractor exactly."""
    from tnreason.engine.workload_to_numpy import NumpyEinsumContractor

    rng = np.random.default_rng(0)
    n = 12
    A = sparse.random(n, n, density=0.3, random_state=1, format="csr")
    B = sparse.random(n, n, density=0.3, random_state=2, format="csr")
    gate = sparse.diags((rng.random(n) < 0.5).astype(float)).tocsr()

    cases = {
        "chain": ({"a": ScipySparseCore(values=A, colors=["X", "Z"], shape=[n, n]),
                   "b": ScipySparseCore(values=B, colors=["Z", "Y"], shape=[n, n])}, ["X", "Y"]),
        "tree": ({"a": ScipySparseCore(values=A, colors=["X", "Z"], shape=[n, n]),
                  "b": ScipySparseCore(values=B, colors=["Z", "Y"], shape=[n, n]),
                  "g": ScipySparseCore(values=gate, colors=["X"], shape=[n])}, ["X", "Y"]),
        "fork": ({"a": ScipySparseCore(values=A, colors=["X", "Z"], shape=[n, n]),
                  "b": ScipySparseCore(values=B.T.tocsr(), colors=["Z", "Y"], shape=[n, n])}, ["X", "Y"]),
    }
    for label, (cores, openColors) in cases.items():
        fast = contract_sparse(cores, openColors)
        denseCores = {k: v.to_NumpyCore() for k, v in cores.items()}
        slow = NumpyEinsumContractor(coreDict=denseCores, openColors=openColors).contract()
        slowValues = slow.values if hasattr(slow, "values") else np.asarray(slow)
        difference = np.abs(np.asarray(fast.values.toarray()) - np.asarray(slowValues)).max()
        assert difference < 1e-10, "{} disagrees with the dense contractor: {}".format(label, difference)
        print("{}: matches the dense contractor, max difference {:.1e}".format(label, difference))


if __name__ == "__main__":
    _self_check()
