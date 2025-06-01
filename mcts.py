import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def uct_score(self, exploration=1.41):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def is_fully_expanded(self, moves):
        return len(self.children) == len(moves)


class MCTSGuidedTree:
    def __init__(self, x, n, p, distances, capacity, demands):
        self.x = x
        self.n = n
        self.p = p
        self.distances = distances
        self.depot = 0
        self.max_capacity = capacity
        self.demands = demands
        self.return_to_depot_prob = n / p
        self.min_value = 1 / (n * p)

        self.best_score = float("inf")
        self.best_paths = []

    def solve(self, num_simulations=10000):
        root = MCTSNode(([], set(), self.max_capacity))
        for _ in range(num_simulations):
            node = self._select(root)
            if node is None:
                continue
            child = self._expand(node)
            if child is None:
                continue
            score, final_paths = self._simulate(child)
            self._backpropagate(child, score)
            if score < self.best_score:
                self.best_score = score
                self.best_paths = final_paths
        return self.best_score, self.best_paths

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.uct_score())
        return node

    def _expand(self, node):
        paths, visited, capacity = node.state
        new_paths = [list(path) for path in paths]
        if not new_paths or new_paths[-1][-1] == self.depot:
            if len(new_paths) >= self.p:
                return None
            new_paths.append([self.depot])
            capacity = self.max_capacity

        path = new_paths[-1]
        current = path[-1]

        possible_moves = []
        for i in range(self.n):
            if i != current and i not in visited and capacity >= self.demands[i]:
                possible_moves.append(i)
        if not possible_moves:
            possible_moves = [self.depot]

        for move in possible_moves:
            if all(child.move != move for child in node.children):
                new_path = list(path)
                new_path.append(move)
                new_paths[-1] = new_path
                new_visited = set(visited)
                if move != self.depot:
                    new_visited.add(move)
                new_capacity = capacity - (
                    self.demands[move] if move != self.depot else 0
                )
                new_node = MCTSNode(
                    (new_paths, new_visited, new_capacity), parent=node, move=move
                )
                node.children.append(new_node)
                return new_node
        return random.choice(node.children)

    def _simulate(self, node):
        paths, visited, capacity = (
            [list(p) for p in node.state[0]],
            set(node.state[1]),
            node.state[2],
        )
        while True:
            if len(paths) == self.p and paths[-1][-1] == self.depot:
                break
            if not paths or paths[-1][-1] == self.depot:
                if len(paths) >= self.p:
                    break
                paths.append([self.depot])
                capacity = self.max_capacity
            path = paths[-1]
            current = path[-1]
            current_vehicle = len(paths) - 1

            moves, weights = [], []
            for i in range(self.n):
                if i != current and i not in visited and capacity >= self.demands[i]:
                    val = (
                        self.x[(current, i, current_vehicle)].varValue + self.min_value
                    )
                    moves.append(i)
                    weights.append(val)
            if not moves:
                path.append(self.depot)
                continue
            move = random.choices(moves, weights=weights)[0]
            path.append(move)
            visited.add(move)
            capacity -= self.demands[move]

        if sum(len(path) - 2 for path in paths) != self.n - 1:
            return float("inf"), []
        return self._calculate_score(paths), paths

    def _backpropagate(self, node, score):
        while node:
            node.visits += 1
            node.value += -score
            node = node.parent

    def _calculate_score(self, paths):
        score = 0
        for path in paths:
            for i in range(len(path) - 1):
                score += self.distances[path[i]][path[i + 1]]
        return score
