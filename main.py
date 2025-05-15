from cvrp import CVRP

if __name__ == "__main__":
    cvrp = CVRP.from_file("data/E-n23-k3.vrp")
    value, paths = cvrp.solve()
    print(cvrp)
    print(f"Minimal tour: {value}")
    print(f"Routes: {paths}")
    cvrp.plot_solution(paths)
