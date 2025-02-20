import math

def minimax(node, depth, alpha, beta, maximizingPlayer):
    # Step-02: Terminating condition
    if depth == 0 or isinstance(node, int):  # Assuming a leaf node is an integer value
        return node

    # Step-03: Maximizer Player Steps
    if maximizingPlayer:
        maxEva = -math.inf
        for child in node:
            eva = minimax(child, depth - 1, alpha, beta, False)
            maxEva = max(maxEva, eva)
            alpha = max(alpha, maxEva)
            if beta <= alpha:
                break  # Beta cut-off
        return maxEva

    # Step-04: Minimizer Player Steps
    else:
        minEva = math.inf
        for child in node:
            eva = minimax(child, depth - 1, alpha, beta, True)
            minEva = min(minEva, eva)
            beta = min(beta, minEva)
            if beta <= alpha:
                break  # Alpha cut-off
        return minEva

def build_tree(flat_tree, depth):
    """Convert a flat array into a nested tree structure."""
    if depth == 0:
        return flat_tree.pop(0)  # Leaf node
    children = []
    for _ in range(2):  # Binary tree: 2 children per node
        children.append(build_tree(flat_tree, depth - 1))
    return children

# Get user input
flattened_tree = list(map(int, input("Enter the flattened game tree (space-separated): ").split()))
depth = int(input("Enter the depth of the tree: "))

# Build the game tree from the flattened input
game_tree = build_tree(flattened_tree, depth)

# Call the minimax algorithm
evaluated_value = minimax(game_tree, depth, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)

print("Evaluated Value:", evaluated_value)