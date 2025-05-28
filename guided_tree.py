import random


class GuidedTree:
    def __init__(self, x, n, p, distances, capacity, demands):
        self.x = x
        self.n = n
        self.p = p
        self.paths = []
        self.distances = distances
        self.visited = set()
        self.depot = 0
        self.counter = 0
        self.best_score = 1e9
        self.best_paths = []
        self.return_to_depot_prob = n / p
        self.max_capacity = capacity
        self.capacity = capacity
        self.demands = demands
        self.min_value = 1 / (n * p)

    def solve(self, num_simulations=1_000_000):
        while self.counter < num_simulations:
            while not self._is_finished():
                self._do_move()
            if self._is_valid():
                score = self._calculate_score()
                if score < self.best_score:
                    print(f"Found new best solution: {score}")
                    self.best_score = score
                    self.best_paths = [path[:] for path in self.paths]
            self._reset()

        return self.best_score, self.best_paths

    def _do_move(self):
        self.counter += 1

        if len(self.paths) == 0 or self.paths[-1][-1] == self.depot:
            self.paths.append([self.depot])
            self.capacity = self.max_capacity
        path = self.paths[-1]
        current_node = path[-1]
        current_vehicle = len(self.paths) - 1
        possible_moves = []
        move_values = []
        for i in range(self.n):
            if (
                i != current_node
                # and self.x[(current_node, i, current_vehicle)].varValue > 0
                and i not in self.visited
                and self.capacity >= self.demands[i]
            ):
                possible_moves.append(i)
                if i == self.depot:
                    move_values.append(
                        max(
                            self.x[(current_node, i, current_vehicle)].varValue,
                            self.return_to_depot_prob,
                        )
                    )
                else:
                    move_values.append(
                        self.x[(current_node, i, current_vehicle)].varValue
                        + self.min_value
                    )

        if len(possible_moves) == 0:
            self.paths[-1].append(self.depot)
        else:
            total_value = sum(move_values)
            random_value = random.uniform(0, total_value)
            move = None
            for i, move_value in enumerate(move_values):
                if random_value < move_value:
                    move = possible_moves[i]
                    break
                random_value -= move_value
            if move != self.depot:
                self.visited.add(move)
            self.capacity -= self.demands[move]
            self.paths[-1].append(move)

    def _reset(self):
        self.paths = []
        self.visited = set()
        self.capacity = self.max_capacity

    def _is_finished(self):
        if len(self.paths) == self.p and self.paths[-1][-1] == self.depot:
            return True

    def _is_valid(self):
        if sum(len(path) - 2 for path in self.paths) == self.n - 1:
            return True
        return False

    def _calculate_score(self):
        score = 0

        for path in self.paths:
            current_node = path[0]
            for next_node in path[1:]:
                score += self.distances[current_node][next_node]
                current_node = next_node
        return score
