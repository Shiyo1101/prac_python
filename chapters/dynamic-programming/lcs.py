from typing import List, Tuple


def lcs_dp(s: str, t: str) -> Tuple[int, List[List[int]]]:
    """
    2つの文字列の最長共通部分列（LCS）の長さと動的計画法（DP）テーブルを計算します。

    Args:
        s: 一方の文字列
        t: もう一方の文字列

    Returns:
        タプル: (LCSの長さ, 計算に使用したDPテーブル)
    """
    n, m = len(s), len(t)

    # DPテーブルを0で初期化 (サイズ: (n+1) x (m+1))
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # DPテーブルを埋める
    for i in range(n):
        for j in range(m):
            # 文字が一致する場合
            if s[i] == t[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            # 文字が一致しない場合
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    return dp[n][m], dp


def lcs_from_table(s: str, t: str, dp: List[List[int]]) -> str:
    """
    DPテーブルから最長共通部分列（LCS）を復元します。

    Args:
        s: lcs_dpで使用した一方の文字列
        t: lcs_dpで使用したもう一方の文字列
        dp: lcs_dpで計算されたDPテーブル

    Returns:
        最長共通部分列（複数ある場合はそのうちの1つ）
    """
    n, m = len(s), len(t)
    lcs = []

    # DPテーブルを右下から左上に向かって逆にたどる
    while n > 0 and m > 0:
        # 文字が一致する場合、その文字はLCSの一部
        if s[n - 1] == t[m - 1]:
            lcs.append(s[n - 1])
            n -= 1
            m -= 1
        # 文字が一致しない場合、値が大きい方に移動する
        elif dp[n - 1][m] >= dp[n][m - 1]:
            n -= 1
        else:
            m -= 1

    # 後ろからLCSを構築したため、最後に反転させる
    lcs.reverse()
    return "".join(lcs)


def print_dp_table(s: str, t: str, dp: List[List[int]]) -> None:
    """
    計算されたDPテーブルを、文字列をヘッダーとして整形して表示します。

    Args:
        s: 一方の文字列（行ヘッダーになります）
        t: もう一方の文字列（列ヘッダーになります）
        dp: 表示するDPテーブル
    """
    n = len(s)
    m = len(t)

    # 列ヘッダー (文字列t) を表示
    # ø は空文字列を表す記号です
    print(f"{'':>5}", end="")
    print(f"{'ø':>3}", end="")
    for char_t in t:
        print(f"{char_t:>3}", end="")
    print("\n" + "-" * (4 + (m + 2) * 3))  # 区切り線

    # 1行目 (sが空文字列の場合) の表示
    print(f"{'ø':>3} |", end="")
    for j in range(m + 1):
        print(f"{dp[0][j]:>3}", end="")
    print()

    # 2行目以降の表示
    for i in range(n):
        # 行ヘッダー (文字列sの文字) を表示
        print(f"{s[i]:>3} |", end="")
        # DPテーブルの値を表示
        for j in range(m + 1):
            print(f"{dp[i + 1][j]:>3}", end="")
        print()


# --- 実行例 ---
if __name__ == "__main__":
    s1 = "AGGTAB"
    s2 = "GXTXAYB"

    lcs_length, dp_table = lcs_dp(s1, s2)
    lcs_string = lcs_from_table(s1, s2, dp_table)

    print(f"文字列1: {s1}")
    print(f"文字列2: {s2}")
    print(f"最長共通部分列の長さ: {lcs_length}")
    print(f"最長共通部分列: {lcs_string}")

    print("\n--- DPテーブル ---")
    print_dp_table(s1, s2, dp_table)
