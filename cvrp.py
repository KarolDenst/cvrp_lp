import re
import math
import pulp
import matplotlib.pyplot as plt
import itertools


class CVRP:
    def __init__(
        self,
        name,
        comment,
        problem_type,
        dimension,
        edge_weight_type,
        capacity,
        node_coords,
        demands,
        depots,
        num_trucks: int,
        optimal_value: int,
    ):
        self.name = name
        self.comment = comment
        self.problem_type = problem_type
        self.dimension = dimension
        self.edge_weight_type = edge_weight_type
        self.capacity = capacity
        self.node_coords = node_coords
        self.demands = demands
        self.depots = depots
        self.num_trucks = num_trucks
        self.optimal_value = optimal_value
        self.distances = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        distances = [[0] * self.dimension for _ in range(self.dimension)]
        node_ids = list(self.node_coords.keys())
        for i in node_ids:
            for j in node_ids:
                if i == j:
                    distances[i][j] = 0
                else:
                    coord_i = self.node_coords[i]
                    coord_j = self.node_coords[j]
                    dist = math.sqrt(
                        (coord_i[0] - coord_j[0]) ** 2 + (coord_i[1] - coord_j[1]) ** 2
                    )
                    distances[i][j] = dist
        return distances

    def __str__(self):
        return (
            f"CVRP Instance: {self.name}\n"
            f"  Comment: {self.comment}\n"
            f"  Type: {self.problem_type}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Edge Weight Type: {self.edge_weight_type}\n"
            f"  Capacity: {self.capacity}\n"
            f"  Node Coords (first 3): {self.node_coords}\n"
            f"  Demands (first 3): {self.demands}\n"
            f"  Depots: {self.depots}\n"
            f"  Num Trucks: {self.num_trucks}\n"
            f"  Optimal Value: {self.optimal_value}"
        )

    @staticmethod
    def from_file(file_path):
        parsed_data = {
            "NAME": None,
            "COMMENT": None,
            "TYPE": None,
            "DIMENSION": None,
            "EDGE_WEIGHT_TYPE": None,
            "CAPACITY": None,
            "NODE_COORD_SECTION": {},
            "DEMAND_SECTION": {},
            "DEPOT_SECTION": [],
        }
        current_section = None
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("COMMENT"):
                    if parsed_data["COMMENT"] is None:
                        match = re.match(r"COMMENT\s*:(.*)", line, re.IGNORECASE)
                        if match:
                            parsed_data["COMMENT"] = match.group(1).strip()
                    continue
                if ":" in line:
                    parts = re.split(r"\s*:\s*", line, 1)
                    if len(parts) == 2:
                        key, value = map(str.strip, parts)
                        key = key.upper().replace(" ", "_")
                        if key in parsed_data:
                            if key in ["DIMENSION", "CAPACITY"]:
                                try:
                                    parsed_data[key] = int(value)
                                except ValueError:
                                    print(
                                        f"Warning: Expected int for {key}, got '{value}'"
                                    )
                                    parsed_data[key] = value
                            else:
                                parsed_data[key] = value
                        else:
                            parsed_data[key] = value  # Store unknown keys too
                        current_section = None
                        continue
                header_map = {
                    "NODE_COORD_SECTION": "NODE_COORD_SECTION",
                    "DEMAND_SECTION": "DEMAND_SECTION",
                    "DEPOT_SECTION": "DEPOT_SECTION",
                }
                if line.upper() in header_map:
                    current_section = header_map[line.upper()]
                elif line.upper() == "EOF":
                    break
                elif current_section:
                    parts = re.split(r"\s+", line)
                    if not parts or not parts[0]:
                        continue
                    try:
                        if current_section == "NODE_COORD_SECTION" and len(parts) >= 3:
                            parsed_data["NODE_COORD_SECTION"][int(parts[0])] = (
                                float(parts[1]),
                                float(parts[2]),
                            )
                        elif current_section == "DEMAND_SECTION" and len(parts) >= 2:
                            parsed_data["DEMAND_SECTION"][int(parts[0])] = int(parts[1])
                        elif current_section == "DEPOT_SECTION":
                            node_id = int(parts[0])
                            if node_id == -1:
                                current_section = None
                            else:
                                parsed_data["DEPOT_SECTION"].append(node_id)
                    except (IndexError, ValueError) as e:
                        print(
                            f"Warning: Skipping malformed line in {current_section}: {line} ({e})"
                        )

        num_trucks, optimal_value = 99999, 999999
        if parsed_data.get("COMMENT"):
            truck_match = re.search(
                r"No of trucks:\s*(\d+)", parsed_data["COMMENT"], re.IGNORECASE
            )
            opt_match = re.search(
                r"Optimal value:\s*(\d+)", parsed_data["COMMENT"], re.IGNORECASE
            )
            if truck_match:
                num_trucks = int(truck_match.group(1))
            if opt_match:
                optimal_value = int(opt_match.group(1))

        if num_trucks is None and parsed_data.get("NAME"):
            name_k_match = re.search(r"-k(\d+)", parsed_data["NAME"])
            if name_k_match:
                num_trucks = int(name_k_match.group(1))
            else:
                num_trucks = 9999

        return CVRP(
            name=parsed_data["NAME"],
            comment=parsed_data["COMMENT"],
            problem_type=parsed_data["TYPE"],
            dimension=parsed_data["DIMENSION"],
            edge_weight_type=parsed_data["EDGE_WEIGHT_TYPE"],
            capacity=parsed_data["CAPACITY"],
            node_coords=parsed_data["NODE_COORD_SECTION"],
            demands=parsed_data["DEMAND_SECTION"],
            depots=parsed_data["DEPOT_SECTION"],
            num_trucks=num_trucks,
            optimal_value=optimal_value,
        )

    def plot_solution(self, paths: list):
        plt.figure(figsize=(10, 8))

        all_node_x = [self.node_coords[node_id][0] for node_id in self.node_coords]
        all_node_y = [self.node_coords[node_id][1] for node_id in self.node_coords]
        plt.scatter(all_node_x, all_node_y, c="blue", label="Customers", s=50)

        # Highlight depots
        depot_x = [
            self.node_coords[depot_id][0]
            for depot_id in self.depots
            if depot_id in self.node_coords
        ]
        depot_y = [
            self.node_coords[depot_id][1]
            for depot_id in self.depots
            if depot_id in self.node_coords
        ]
        plt.scatter(depot_x, depot_y, c="red", marker="s", s=100, label="Depot(s)")

        # Add node labels (optional, can be cluttered)
        for node_id, (x, y) in self.node_coords.items():
            plt.text(x, y + 0.5, str(node_id), fontsize=9)

        # Plot paths
        colors = plt.cm.get_cmap(
            "tab10", len(paths)
        )  # Get a distinct color for each path

        for i, path in enumerate(paths):
            path_x = []
            path_y = []
            for node_index in path:
                coord = self.node_coords[node_index]
                path_x.append(coord[0])
                path_y.append(coord[1])

            if path_x:  # If any valid coords were added
                plt.plot(
                    path_x,
                    path_y,
                    marker="o",
                    linestyle="-",
                    color=colors(i),
                    label=f"Vehicle {i + 1}",
                )

        plt.title(f"CVRP Solution: {self.name}")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def solve(self) -> tuple[float, list[int]]:
        prob = pulp.LpProblem("CVRP", pulp.LpMinimize)
        p = self.num_trucks
        n = self.dimension
        customer_nodes = [i for i in range(n) if i not in self.depots]

        # Variables
        indices = [(i, j, k) for i in range(n) for j in range(n) for k in range(p)]
        x = pulp.LpVariable.dicts(
            name="x", indices=indices, cat=pulp.LpBinary, lowBound=0, upBound=1
        )

        # Goal function
        prob += pulp.lpSum(self.distances[i][j] * x[(i, j, k)] for i, j, k in indices)

        # Constraints
        # No traveling from itself to itself
        for k in range(p):
            for i in range(n):
                prob += x[(i, i, k)] == 0

        # Vehicle leaves node that it enters
        for k in range(p):
            for i in range(n):
                prob += pulp.lpSum(x[(i, j, k)] for j in range(n)) == pulp.lpSum(
                    x[(j, i, k)] for j in range(n)
                )

        # Ensure that every node is entered once
        for j in customer_nodes:
            prob += pulp.lpSum(x[(i, j, k)] for i in range(n) for k in range(p)) == 1

        # Every vehicle leaves the depot
        # Not sure if this is required

        # Capacity constraint
        for k in range(p):
            prob += (
                pulp.lpSum(
                    self.demands[j] * x[(i, j, k)]
                    for i in range(n)
                    for j in customer_nodes
                )
                <= self.capacity
            )

        # Eliminate subtours
        for subset_size in range(2, n - 1):
            for subset in itertools.combinations(customer_nodes, subset_size):
                subset_nodes = set(subset)
                prob += (
                    pulp.lpSum(
                        x[(i, j, k)]
                        for i in subset_nodes
                        for j in range(n)
                        if j not in subset_nodes
                        for k in range(p)
                    )
                    >= 1
                )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        paths = self._reconstruct_paths(x, n, p)

        return pulp.value(prob.objective), paths

    def _reconstruct_paths(self, x, n, p) -> list[int]:
        def dfs(node, k, visited):
            path = [node]
            for j in range(n):
                if (
                    (node, j, k) in x
                    and x[(node, j, k)].varValue == 1
                    and (node, j) not in visited
                ):
                    visited.add((node, j))
                    print(node, j, k, x[(node, j, k)].name, x[(node, j, k)].varValue)
                    path.extend(dfs(j, k, visited))
                    break
            return path

        paths = []
        for k in range(p):
            visited = set()
            for depot in self.depots:
                path = dfs(depot, k, visited)
                if len(path) > 1:
                    paths.append(path)
                    break

        return paths
