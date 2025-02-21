import math

def minimax(node, depth, alpha, beta, maximizing):
    if depth == 0 or not isinstance(node, list):  
        return node  # Leaf node

    best = -math.inf if maximizing else math.inf

    for child in node:
        val = minimax(child, depth-1, alpha, beta, not maximizing)
        best = max(best, val) if maximizing else min(best, val)
        alpha, beta = (max(alpha, best), beta) if maximizing else (alpha, min(beta, best))
        if beta <= alpha:
            break  # Alpha-beta pruning

    return best

def build_tree(values, depth):
    """Builds a binary tree from user input."""
    if depth == 0:
        return values.pop(0) if values else 0
    return [build_tree(values, depth-1), build_tree(values, depth-1)]

# User input
values = list(map(int, input("Enter leaf node values (space-separated): ").split()))
depth = int(input("Enter tree depth: "))

# Build tree and run minimax
tree = build_tree(values, depth)
print("Best value:", minimax(tree, depth, -math.inf, math.inf, True))
