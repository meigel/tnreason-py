"""Step 1: boosting loop over chain architectures with filtered link prediction.

Sparse form of contracting the selection architecture against the data: candidate bodies are
priced against every head relation in one pass over the fact keys, and rules are added
greedily under a residual objective, the sparse evaluation of ProposalDistribution's
negative phase.

Fixes over the first version, following the diagnosis in results_step1.md:
  * candidate bodies come from the full relation alphabet, pruned by the necessary condition
    that body domain/range intersect the head's (contraction-order argument, cheapest form)
  * bodies may carry three premises, extended from the surviving two-premise bodies
  * rule scores aggregate by maximum firing confidence instead of a sum
  * the residual stopping rule is a parameter (minSupport) rather than a constant

Usage:
    python boost.py --selftest
    python boost.py --depth 3 --rules 20 --minSupport 2 --aggregation max
"""

import argparse
import os
import time

import numpy as np

import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def encode(rows, cols, n):
    return rows.astype(np.int64) * n + cols.astype(np.int64)


def sorted_keys(matrix, n):
    """Sorted encoded keys of the nonzeros. Matmul returns non-canonical index order."""
    counts = np.diff(matrix.indptr)
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), counts)
    return np.sort(encode(rows, matrix.indices, n))


def build_fact_index(rows, cols, rIdx, n):
    keys = encode(rows, cols, n)
    order = np.argsort(keys, kind="stable")
    return keys[order], rIdx[order]


def domains_ranges(cores):
    dom, rang = [], []
    for core in cores:
        rows = np.repeat(np.arange(core.shape[0], dtype=np.int64), np.diff(core.indptr))
        dom.append(np.unique(rows))
        rang.append(np.unique(core.indices.astype(np.int64)))
    return dom, rang


def overlaps(target, others):
    """Boolean vector: does `target` intersect each array in `others`."""
    out = np.zeros(len(others), dtype=bool)
    if target.size == 0:
        return out
    for i, other in enumerate(others):
        if other.size == 0:
            continue
        pos = np.minimum(np.searchsorted(other, target), other.size - 1)
        out[i] = bool((other[pos] == target).any())
    return out


def stride(items, limit):
    if len(items) <= limit:
        return items
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


def chain(cores, body):
    out = cores[body[0]]
    for k in body[1:]:
        out = out @ cores[k]
    return out


def price_bodies(cores, n, factKeys, factRels, nRel, bodies):
    """support[head, body] = head facts (x,y) for which the body holds for some intermediates."""
    support = np.zeros((nRel, len(bodies)), dtype=np.float32)
    bodyNnz = np.zeros((nRel, len(bodies)), dtype=np.float32)
    paths = {}
    t0 = time.time()
    for k, body in enumerate(bodies):
        path = chain(cores, body)
        if path.nnz == 0:
            continue
        paths[k] = path
        pkeys = sorted_keys(path, n)
        pos = np.minimum(np.searchsorted(pkeys, factKeys), pkeys.size - 1)
        hit = pkeys[pos] == factKeys
        support[:, k] = np.bincount(factRels[hit], minlength=nRel)
        bodyNnz[:, k] = path.nnz
    print("  priced {} bodies in {:.1f}s".format(len(bodies), time.time() - t0), flush=True)
    return support, bodyNnz, paths


def learn_rules(entities, relations, rows, cols, rIdx, depth, ruleCount, minSupport, maxBodies=4000,
                inverse=False):
    n, nRel = len(entities), len(relations)
    headCores = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = headCores + [c.T.tocsr() for c in headCores] if inverse else headCores
    nAtom = len(cores)
    factKeys, factRels = build_fact_index(rows, cols, rIdx, n)
    dom, rang = domains_ranges(cores)

    domOverlap = np.array([overlaps(dom[h], dom) for h in range(nRel)])
    rangOverlap = np.array([overlaps(rang[h], rang) for h in range(nRel)])
    joint = np.array([overlaps(rang[a], dom) for a in range(nAtom)])

    bodies = [(a, b) for a in range(nAtom) for b in range(nAtom) if joint[a, b]]
    print("{} of {} two-premise bodies pass domain/range pruning over {} atoms".format(
        len(bodies), nAtom ** 2, nAtom), flush=True)
    # ponytail: uniform stride keeps every relation index range represented; switch to
    # support-based pruning only if the stride measurably loses rules
    bodies = stride(bodies, maxBodies)
    support, bodyNnz, paths = price_bodies(cores, n, factKeys, factRels, nRel, bodies)
    allBodies = list(bodies)

    if depth >= 3:
        extensions = set()
        for h in range(nRel):
            viable = [k for k, (a, b) in enumerate(allBodies) if domOverlap[h, a] and rangOverlap[h, b]]
            viable = sorted(viable, key=lambda k: -support[h, k])[:5]
            for k in viable:
                if support[h, k] < minSupport:
                    continue
                a, b = allBodies[k]
                for c in range(nAtom):
                    if joint[b, c]:
                        extensions.add((a, b, c))
        extensions = stride(sorted(extensions), maxBodies)
        print("{} three-premise bodies after extension and pruning".format(len(extensions)), flush=True)
        if extensions:
            s3, n3, p3 = price_bodies(cores, n, factKeys, factRels, nRel, extensions)
            start = len(allBodies)
            allBodies.extend(extensions)
            for k in p3:                      # bodies with empty paths keep their column
                paths[start + k] = p3[k]
            support = np.hstack([support, s3])
            bodyNnz = np.hstack([bodyNnz, n3])

    headKeys = [np.sort(encode(rows[rIdx == h], cols[rIdx == h], n)) for h in range(nRel)]
    candKeys = {k: sorted_keys(p, n) for k, p in paths.items()}

    ruleSets, shortlistSize = {}, max(60, 4 * ruleCount)
    for h in range(nRel):
        ruleSets[h] = []
        if support[h].max() < minSupport:
            continue
        keys = headKeys[h]
        hitsOf = {}
        for k in np.argsort(-support[h])[:shortlistSize]:
            k = int(k)
            if support[h, k] < minSupport or k not in candKeys:
                continue
            pos = np.minimum(np.searchsorted(candKeys[k], keys), candKeys[k].size - 1)
            hitsOf[k] = np.nonzero(candKeys[k][pos] == keys)[0]
        covered = np.zeros(keys.size, dtype=bool)
        for _ in range(ruleCount):
            best = None
            for k, hits in hitsOf.items():
                if any(rule["cand"] == k for rule in ruleSets[h]):
                    continue
                residual = int((~covered[hits]).sum())
                if residual >= minSupport and (best is None or residual > best[0]):
                    best = (residual, k)
            if best is None:
                break
            residual, k = best
            covered[hitsOf[k]] = True
            ruleSets[h].append({"cand": k, "body": allBodies[k], "support": float(support[h, k]),
                                "confidence": float(support[h, k] / max(float(bodyNnz[h, k]), 1.0)),
                                "residual": residual})
        if ruleSets[h]:
            top = ruleSets[h][0]
            print("{:<42} rules={} conf={:.3f} depth={} body={} support={:g}".format(
                relations[h][:42], len(ruleSets[h]), top["confidence"], len(top["body"]),
                ".".join(atomName(relations, r, nRel) for r in top["body"]), top["support"]),
                flush=True)
    total = sum(len(v) for v in ruleSets.values())
    print("learned {} rules over {} head relations".format(total, nRel), flush=True)
    return cores, ruleSets, allBodies, n, total


def atomName(relations, atom, nRel):
    return relations[atom] if atom < nRel else "inv(" + relations[atom - nRel] + ")"


def read_triples(path, entIndex, relIndex):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[0] in entIndex and parts[2] in entIndex and parts[1] in relIndex:
                out.append((entIndex[parts[0]], relIndex[parts[1]], entIndex[parts[2]]))
    return out


def evaluate(cores, ruleSets, n, relations, entIndex, relIndex, trainIdx, extraPaths, testPath,
             maxTest=None, priorOnly=False, aggregation="max"):
    known = {}
    for h, r, t in trainIdx:
        known.setdefault((h, r), set()).add(t)
    for path in extraPaths:
        for h, r, t in read_triples(path, entIndex, relIndex):
            known.setdefault((h, r), set()).add(t)
    test = read_triples(testPath, entIndex, relIndex)
    if maxTest:
        test = test[:maxTest]
    for h, r, t in test:
        known.setdefault((h, r), set()).add(t)

    byHead = {}
    for h, r, t in test:
        byHead.setdefault(r, []).append((h, t))

    tailCounts = np.zeros((len(relations), n), dtype=np.float32)
    np.add.at(tailCounts, (np.array([r for _, r, _ in trainIdx], dtype=np.int64),
                           np.array([t for _, _, t in trainIdx], dtype=np.int64)), 1.0)

    ranks, positives, t0 = [], [], time.time()
    for hIdx in sorted(byHead):
        rules = ruleSets[hIdx]
        if not rules:
            continue
        score = None
        for rule in ([] if priorOnly else rules):
            path = chain(cores, rule["body"])
            term = (path > 0).astype(np.float32) * rule["confidence"]
            if score is None:
                score = term
            elif aggregation == "max":
                score = score.maximum(term)
            else:
                score = score + term
        pairs = byHead[hIdx]
        targetRows = np.array(sorted({x for x, _ in pairs}))
        sub = np.zeros((targetRows.size, n), dtype=np.float64) if score is None \
            else score[targetRows, :].toarray().astype(np.float64)
        # ties broken by the head relation's empirical tail marginal, then by entity index:
        # score 0 means no rule fired, which is not evidence of a wrong answer
        prior = tailCounts[hIdx] / max(float(tailCounts[hIdx].max()), 1.0)
        sub = sub + 1e-6 * prior[None, :] - 1e-9 * np.arange(n, dtype=np.float64)[None, :]
        rowOf = {int(x): i for i, x in enumerate(targetRows)}
        for x, y in pairs:
            row = sub[rowOf[x]]
            trueTails = np.array([t for t in known[(x, hIdx)] if t != y], dtype=np.int64)
            target = row[y]
            if trueTails.size:
                row = row.copy()
                row[trueTails] = -np.inf
            ranks.append(1 + int((row > target).sum()))
            positives.append(int((row > 0).sum()))
    ranks = np.array(ranks, dtype=np.float64)
    positives = np.array(positives, dtype=np.float64)
    if ranks.size == 0:
        print("no test triple covered by any rule", flush=True)
        return None
    print("\nfiltered evaluation on {} of {} test triples ({:.1f}s), aggregation={}".format(
        ranks.size, len(test), time.time() - t0, aggregation), flush=True)
    print("mean scored candidates per query: {:.2f} of {}".format(positives.mean(), n), flush=True)
    print("MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
        float((1.0 / ranks).mean()), float((ranks <= 1).mean()),
        float((ranks <= 3).mean()), float((ranks <= 10).mean())), flush=True)
    return float((1.0 / ranks).mean())


def selftest():
    """Planted composition c = a o b must be recovered with confidence one."""
    rng = np.random.default_rng(1)
    n = 200

    def functional(k=2):
        m = np.zeros((n, n), dtype=bool)
        m[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
        return m

    a, b = functional(), functional()
    c = (a.astype(np.int32) @ b.astype(np.int32)) > 0
    rows, cols, rIdx = [], [], []
    for mat, rel in ((a, 0), (b, 1), (c, 2)):
        xs, ys = np.nonzero(mat)
        rows.extend(xs.tolist())
        cols.extend(ys.tolist())
        rIdx.extend([rel] * len(xs))
    rows, cols, rIdx = np.array(rows), np.array(cols), np.array(rIdx)
    entities = ["e" + str(i) for i in range(n)]
    relations = ["a", "b", "c"]
    cores, ruleSets, bodies, _, _ = learn_rules(entities, relations, rows, cols, rIdx,
                                                depth=2, ruleCount=2, minSupport=3)
    assert ruleSets[2], "no rule learned for the planted head relation"
    body = ruleSets[2][0]["body"]
    print("selftest learned body:", [relations[r] for r in body], flush=True)
    assert body == (0, 1), "planted body not recovered, got {}".format(body)
    assert abs(ruleSets[2][0]["confidence"] - 1.0) < 1e-6, "confidence of an exact composition must be one"
    print("selftest passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--minSupport", type=int, default=2)
    parser.add_argument("--maxBodies", type=int, default=4000)
    parser.add_argument("--aggregation", default="max", choices=["max", "sum"])
    parser.add_argument("--maxTest", type=int, default=None)
    parser.add_argument("--priorOnly", action="store_true")
    parser.add_argument("--inverse", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        trainPath = os.path.join(args.dir, "train.txt")
        entities, relations, rows, cols, rIdx = K.load_triples(trainPath)
        entIndex = {e: i for i, e in enumerate(entities)}
        relIndex = {r: i for i, r in enumerate(relations)}
        cores, ruleSets, bodies, n, total = learn_rules(entities, relations, rows, cols, rIdx,
                                                        args.depth, args.rules, args.minSupport,
                                                        args.maxBodies, args.inverse)
        trainIdx = list(zip(rows.tolist(), rIdx.tolist(), cols.tolist()))
        evaluate(cores, ruleSets, n, relations, entIndex, relIndex, trainIdx,
                 [os.path.join(args.dir, "valid.txt")], os.path.join(args.dir, "test.txt"),
                 args.maxTest, args.priorOnly, args.aggregation)
