"""CLI：建索引嵌入性能探测。"""

import argparse

from rag.profile import profile_batch_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile embedding batch sizes")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to test",
    )
    args = parser.parse_args()
    profile_batch_sizes(args.sizes)


if __name__ == "__main__":
    main()
