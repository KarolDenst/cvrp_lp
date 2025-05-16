from cvrp import CVRP

if __name__ == "__main__":
    cvrp = CVRP.from_file("data/simple.vrp")
    value, paths = cvrp.solve(log=True)
    print(cvrp)
    print(f"Minimal tour: {value}")
    print(f"Routes: {paths}")
    cvrp.plot_solution(paths)
