"""Step 1: boosting loop over chain architectures with filtered link prediction.

Rule discovery prices every candidate 2-premise chain against every head relation in
one pass over the fact keys, which is the sparse form of contracting the selection
architecture against the data. Rules are then added greedily under a residual
objective: a rule is scored on head facts that the current rule set does not yet
explain, which is the sparse evaluation of the negative phase in ProposalDistribution.

Usage:
    python boost.py --selftest
    python boost.py --premises 40 --rules 10 --minSupport 5
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
    """Sorted encoded keys of the nonzeros of a csr matrix."""
    counts = np.diff(matrix.indptr)
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), counts)
    return np.sort(encode(rows, matrix.indices, n))


def build_fact_index(rows, cols, rIdx, n):
    keys = encode(rows, cols, n)
    order = np.argsort(keys, kind="stable")
    return keys[order], rIdx[order]


def price_candidates(cores, n, factKeys, factRels, nRel, cands):
    """support[head, cand] = head facts (x,y) for which candidate body holds."""
    support = np.zeros((nRel, len(cands)), dtype=np.float32)
    bodyNnz = np.zeros((nRel, len(cands)), dtype=np.float32)
    paths = {}
    for k, (a, b) in enumerate(cands):
        path = cores[a] @ cores[b]
        if path.nnz == 0:
            continue
        paths[k] = path
        pkeys = sorted_keys(path, n)
        pos = np.minimum(np.searchsorted(pkeys, factKeys), pkeys.size - 1)
        hit = pkeys[pos] == factKeys
        support[:, k] = np.bincount(factRels[hit], minlength=nRel)
        bodyNnz[:, k] = path.nnz
    return support, bodyNnz, paths


def learn_rules(entities, relations, rows, cols, rIdx, premiseCount, ruleCount, minSupport):
    n, nRel = len(entities), len(relations)
    cores = K.relation_cores(rows, cols, rIdx, n, nRel)
    factKeys, factRels = build_fact_index(rows, cols, rIdx, n)

    freqOrder = np.argsort([-int(cores[a].nnz) for a in range(nRel)])[:premiseCount]
    premiseRels = sorted(int(a) for a in freqOrder)
    cands = [(a, b) for a in premiseRels for b in premiseRels]

    t0 = time.time()
    print("pricing {} chains over {} premise relations".format(len(cands), len(premiseRels)), flush=True)
    support, bodyNnz, paths = price_candidates(cores, n, factKeys, factRels, nRel, cands)
    print("priced in {:.1f}s".format(time.time() - t0), flush=True)

    headKeys = [np.sort(encode(rows[rIdx == h], cols[rIdx == h], n)) for h in range(nRel)]
    candKeys = {k: sorted_keys(path, n) for k, path in paths.items()}

    ruleSets = {}
    shortlistSize = max(50, 4 * ruleCount)
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
            ruleSets[h].append({"cand": k, "support": float(support[h, k]),
                                "confidence": float(support[h, k] / max(float(bodyNnz[h, k]), 1.0)),
                                "residual": residual})
        if ruleSets[h]:
            top = ruleSets[h][0]
            print("{:<45} rules={} conf={:.3f} support={:g}".format(
                relations[h][:45], len(ruleSets[h]), top["confidence"], top["support"]), flush=True)
    return cores, ruleSets, cands, n, sum(len(v) for v in ruleSets.values())


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


def evaluate(cores, ruleSets, cands, n, relations, entIndex, relIndex, trainIdx, extraPaths, testPath, maxTest=None, priorOnly=False):
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

    ranks, ranksPess, positives = [], [], []
    tailCounts = np.zeros((len(relations), n), dtype=np.float32)
    np.add.at(tailCounts, (np.array([r for _, r, _ in trainIdx], dtype=np.int64),
                           np.array([t for _, _, t in trainIdx], dtype=np.int64)), 1.0)
    t0 = time.time()
    for hIdx in sorted(byHead):
        rules = ruleSets[hIdx]
        if not rules:
            continue
        score = 0
        for rule in [] if priorOnly else rules:
            a, b = cands[rule["cand"]]
            score = score + (cores[a] @ cores[b]).astype(np.float32) * rule["confidence"]
        pairs = byHead[hIdx]
        targetRows = np.array(sorted({x for x, _ in pairs}))
        sub = np.zeros((targetRows.size, n), dtype=np.float64) if priorOnly else score[targetRows, :].toarray()
        # ties broken by the empirical tail marginal of the head relation: score 0 means no
        # rule fired, which is not evidence of a wrong answer, so it must not win the tie
        prior = tailCounts[hIdx] / max(float(tailCounts[hIdx].max()), 1.0)
        sub = sub.astype(np.float64) + 1e-6 * prior[None, :] - 1e-9 * np.arange(n, dtype=np.float64)[None, :]
        rowOf = {int(x): i for i, x in enumerate(targetRows)}
        for x, y in pairs:
            row = sub[rowOf[x]].copy()
            trueTails = np.array([t for t in known[(x, hIdx)] if t != y], dtype=np.int64)
            target = row[y]
            row[trueTails] = -np.inf
            greater = int((row > target).sum())
            equal = int((row == target).sum()) - 1  # exclude the target itself
            ranks.append(1 + greater)
            ranksPess.append(1 + greater + equal)
            positives.append(int((row > 0).sum()))
    ranks = np.array(ranks, dtype=np.float64)
    ranksPess = np.array(ranksPess, dtype=np.float64)
    positives = np.array(positives, dtype=np.float64)
    if ranks.size == 0:
        print("no test triple covered by any rule", flush=True)
        return None
    print("\nfiltered evaluation on {} of {} test triples ({:.1f}s)".format(
        ranks.size, len(test), time.time() - t0), flush=True)
    print("mean scored candidates per query: {:.2f}".format(positives.mean()), flush=True)
    print("optimistic ties (strictly greater):  MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
        float((1.0 / ranks).mean()), float((ranks <= 1).mean()),
        float((ranks <= 3).mean()), float((ranks <= 10).mean())), flush=True)
    print("pessimistic ties (greater or equal): MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
        float((1.0 / ranksPess).mean()), float((ranksPess <= 1).mean()),
        float((ranksPess <= 3).mean()), float((ranksPess <= 10).mean())), flush=True)
    return float((1.0 / ranksPess).mean())


def selftest():
    """Planted chain c = a o b must be recovered as the rule for c."""
    rng = np.random.default_rng(1)
    n = 200

    def functional(k=2):
        m = np.zeros((n, n), dtype=bool)
        m[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
        return m

    a = functional()
    b = functional()
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
    cores, ruleSets, cands, _, _ = learn_rules(entities, relations, rows, cols, rIdx,
                                               premiseCount=3, ruleCount=2, minSupport=3)
    assert ruleSets[2], "no rule learned for the planted head relation"
    aIdx, bIdx = cands[ruleSets[2][0]["cand"]]
    print("selftest learned body:", relations[aIdx], relations[bIdx], flush=True)
    assert (aIdx, bIdx) == (0, 1), "planted body not recovered, got ({},{})".format(aIdx, bIdx)
    print("selftest passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--premises", type=int, default=40)
    parser.add_argument("--rules", type=int, default=10)
    parser.add_argument("--minSupport", type=int, default=5)
    parser.add_argument("--maxTest", type=int, default=None)
    parser.add_argument("--priorOnly", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        trainPath = os.path.join(args.dir, "train.txt")
        entities, relations, rows, cols, rIdx = K.load_triples(trainPath)
        entIndex = {e: i for i, e in enumerate(entities)}
        relIndex = {r: i for i, r in enumerate(relations)}
        cores, ruleSets, cands, n, totalRules = learn_rules(entities, relations, rows, cols, rIdx,
                                                            args.premises, args.rules, args.minSupport)
        print("learned {} rules over {} head relations".format(totalRules, len(relations)), flush=True)
        trainIdx = list(zip(rows.tolist(), rIdx.tolist(), cols.tolist()))
        evaluate(cores, ruleSets, cands, n, relations, entIndex, relIndex, trainIdx,
                 [os.path.join(args.dir, "valid.txt")], os.path.join(args.dir, "test.txt"), args.maxTest,
                args.priorOnly)
