import pulp
import matplotlib.pyplot as plt
import itertools
import vrplib


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
        distances: list[list[float]],
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
        self.distances = distances

    def __str__(self):
        return (
            f"CVRP Instance: {self.name}\n"
            f"  Comment: {self.comment}\n"
            f"  Type: {self.problem_type}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Edge Weight Type: {self.edge_weight_type}\n"
            f"  Capacity: {self.capacity}\n"
            f"  Node Coords: {self.node_coords}\n"
            f"  Demands: {self.demands}\n"
            f"  Depots: {self.depots}\n"
        )

    @staticmethod
    def from_file(file_path):
        instance_data = vrplib.read_instance(file_path)

        return CVRP(
            name=instance_data["name"],
            comment=instance_data["comment"],
            problem_type=instance_data["type"],
            dimension=instance_data["dimension"],
            edge_weight_type=instance_data["edge_weight_type"],
            capacity=instance_data["capacity"],
            node_coords=instance_data["node_coord"]
            if "node_coord" in instance_data
            else None,
            demands=instance_data["demand"],
            depots=instance_data["depot"],
            distances=instance_data["edge_weight"],
        )

    def plot_solution(self, paths: list):
        if self.node_coords is None or not paths:
            print("No node coordinates or paths to plot.")
            return
        plt.figure(figsize=(10, 8))

        all_node_x = [self.node_coords[node_id][0] for node_id in range(self.dimension)]
        all_node_y = [self.node_coords[node_id][1] for node_id in range(self.dimension)]
        plt.scatter(all_node_x, all_node_y, c="blue", label="Customers", s=50)

        depot_x = [self.node_coords[depot_id][0] for depot_id in self.depots]
        depot_y = [self.node_coords[depot_id][1] for depot_id in self.depots]
        plt.scatter(depot_x, depot_y, c="red", marker="s", s=100, label="Depot(s)")

        for node_id, (x, y) in enumerate(self.node_coords):
            plt.text(x, y + 0.5, str(node_id), fontsize=9)

        colors = plt.cm.get_cmap("tab10", len(paths))

        for i, path in enumerate(paths):
            path_x = []
            path_y = []
            for node_index in path:
                coord = self.node_coords[node_index]
                path_x.append(coord[0])
                path_y.append(coord[1])

            if path_x:
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

    def solve(self, num_trucks, log=False) -> tuple[float, list[int]]:
        prob, x = self._solve_lp(num_trucks=num_trucks, relaxed=False, log=log)
        paths = self._reconstruct_paths_ilp(x, self.dimension, num_trucks)

        return pulp.value(prob.objective), paths

    def _solve_lp(
        self, num_trucks, relaxed=False, log=False
    ) -> tuple[pulp.LpProblem, dict[tuple[int, int, int], pulp.LpVariable]]:
        prob = pulp.LpProblem("CVRP", pulp.LpMinimize)
        p = num_trucks  # The number of trucks is not defined in the vrp file format. Not sure what to put here but it needs to be changed.
        n = self.dimension
        customer_nodes = [i for i in range(n) if i not in self.depots]

        # Variables
        indices = [(i, j, k) for i in range(n) for j in range(n) for k in range(p)]
        x = pulp.LpVariable.dicts(
            name="x",
            indices=indices,
            cat=pulp.LpBinary if not relaxed else pulp.LpContinuous,
            lowBound=0,
            upBound=1,
        )

        # Goal function
        prob += pulp.lpSum(self.distances[i][j] * x[(i, j, k)] for i, j, k in indices)

        self._add_base_constraints(prob, customer_nodes, x, n, p)
        self._eliminate_subtour_constraints(
            prob, customer_nodes, x, n, p, formulation="mtz" if not relaxed else "dfj"
        )

        options = pulp.PULP_CBC_CMD(
            msg=log,
            timeLimit=60,
        )
        prob.solve(options)

        return prob, x

    def _add_base_constraints(self, prob, customer_nodes, x, n, p):
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

    def _eliminate_subtour_constraints(
        self, prob, customer_nodes, x, n, p, formulation="dfj"
    ):
        # DFJ
        if formulation == "dfj":
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
        elif formulation == "mtz":
            n_customers = len(customer_nodes)
            all_nodes = list(range(n))  # All nodes including depots
            u_vars = pulp.LpVariable.dicts(
                "u_seq",  # Name for MTZ auxiliary variables
                [(c_node, k_idx) for c_node in customer_nodes for k_idx in range(p)],
                lowBound=0,  # Can be 0 if c_node is not visited by k_idx
                # upBound=n_customers, # Implicitly handled by constraints below
                cat=pulp.LpContinuous,
            )

            for k_idx in range(p):
                for i_cust_node in customer_nodes:
                    is_i_serviced_by_k = pulp.lpSum(
                        x[(j_node, i_cust_node, k_idx)]
                        for j_node in all_nodes
                        if j_node != i_cust_node
                    )

                    prob += (
                        u_vars[(i_cust_node, k_idx)] >= is_i_serviced_by_k,
                        f"MTZ_LowerBound_u_c{i_cust_node}_k{k_idx}",
                    )
                    prob += (
                        u_vars[(i_cust_node, k_idx)]
                        <= n_customers * is_i_serviced_by_k,
                        f"MTZ_UpperBound_u_c{i_cust_node}_k{k_idx}",
                    )

                    for j_cust_node in customer_nodes:
                        if i_cust_node == j_cust_node:
                            continue
                        prob += (
                            u_vars[(i_cust_node, k_idx)]
                            - u_vars[(j_cust_node, k_idx)]
                            + n_customers * x[(i_cust_node, j_cust_node, k_idx)]
                            <= n_customers - 1,
                            f"MTZ_Order_c{i_cust_node}_c{j_cust_node}_k{k_idx}",
                        )
        else:
            raise ValueError(
                "Unsupported formulation for eliminating subtours. Use 'dfj' or 'mtz'."
            )

    def _reconstruct_paths_ilp(self, x, n, p) -> list[int]:
        def dfs(node, k, visited):
            path = [node]
            for j in range(n):
                if (
                    (node, j, k) in x
                    and x[(node, j, k)].varValue == 1
                    and (node, j) not in visited
                ):
                    visited.add((node, j))
                    path.extend(dfs(j, k, visited))
                    break
            return path

        paths = []
        for k in range(p):
            visited = set()
            for depot in self.depots:
                path = dfs(int(depot), k, visited)
                if len(path) > 1:
                    paths.append(path)
                    break

        return paths
