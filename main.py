import argparse
from cvrp import CVRP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Capacitated Vehicle Routing Problem (CVRP)."
    )
    parser.add_argument("file_path", type=str, help="Path to the .vrp data file.")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging during the solving process. (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        dest="plot",
        help="Enable plotting of the solution. (default: False)",
    )

    args = parser.parse_args()

    cvrp = CVRP.from_file(args.file_path)
    value, paths = cvrp.solve(log=args.log)
    print(cvrp)
    print(f"Minimal tour: {value}")
    print(f"Routes: {paths}")

    if args.plot:
        cvrp.plot_solution(paths)
