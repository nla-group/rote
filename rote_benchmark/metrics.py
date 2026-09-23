"""The two rollout distances reported by ROTE."""


def normalized_damerau_levenshtein_distance(source: str, target: str) -> float:
    """Optimal-string-alignment edit distance divided by the longer length."""
    if source == target:
        return 0.0
    if not source or not target:
        return 1.0
    rows, cols = len(source), len(target)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        dp[i][0] = i
    for j in range(cols + 1):
        dp[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = source[i - 1] != target[j - 1]
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[rows][cols] / max(rows, cols)


def normalized_jaro_winkler_distance(source: str, target: str, p: float = 0.1) -> float:
    """One minus Jaro-Winkler similarity, matching the manuscript runs."""
    if source == target:
        return 0.0
    if not source or not target:
        return 1.0
    rows, cols = len(source), len(target)
    radius = max(rows, cols) // 2 - 1
    matched_source = [False] * rows
    matched_target = [False] * cols
    matches = 0
    for i, symbol in enumerate(source):
        for j in range(max(0, i - radius), min(cols, i + radius + 1)):
            if symbol == target[j] and not matched_target[j]:
                matched_source[i] = matched_target[j] = True
                matches += 1
                break
    if matches == 0:
        return 1.0
    transpositions = 0
    j = 0
    for i, matched in enumerate(matched_source):
        if matched:
            while not matched_target[j]:
                j += 1
            transpositions += source[i] != target[j]
            j += 1
    jaro = (matches / rows + matches / cols + (matches - transpositions // 2) / matches) / 3
    prefix = 0
    for left, right in zip(source[:4], target[:4]):
        if left != right:
            break
        prefix += 1
    return 1 - (jaro + prefix * p * (1 - jaro))
