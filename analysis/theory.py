"""Witness-theory predictions (Propositions 1 to 3 of the plan). Analytic, not measured."""
from math import comb


def random_share(n, k):
    return comb(k, 2) / comb(n, 2)


def tree_worst_share(n, k):
    return max(k - 1, 0) / (n - 1)


def board_share(n, k, readers):
    return comb(k, readers + 1) / comb(n, readers + 1) if k >= readers + 1 else 0.0


if __name__ == "__main__":
    print("k  random  tree_worst  board(r=3)")
    for k in (1, 2, 3, 4, 6, 8, 12):
        print(f"{k:2d} {random_share(12, k):7.3f} {tree_worst_share(12, k):10.3f} {board_share(12, k, 3):10.4f}")
