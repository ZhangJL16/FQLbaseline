import argparse
import os
import sys

from envs.minari_utils import MINARI_BENCHMARK_GROUPS, get_minari_benchmark_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-download the Minari datasets used by this repository."
    )
    parser.add_argument(
        "--group",
        choices=sorted(MINARI_BENCHMARK_GROUPS),
        default="all",
        help="Dataset subset to download.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Explicit Minari dataset ids to download, e.g. D4RL/pen/expert-v2.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected dataset ids and exit.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download datasets even if they already exist locally.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading the remaining datasets if one fails.",
    )
    parser.add_argument(
        "--remote",
        help="Override the Minari remote, e.g. hf://farama-minari.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Override the local Minari dataset cache directory.",
    )
    return parser.parse_args()


def resolve_datasets(args):
    if args.datasets:
        return tuple(args.datasets)
    return get_minari_benchmark_datasets(args.group)


def main():
    args = parse_args()

    if args.remote:
        os.environ["MINARI_REMOTE"] = args.remote
    if args.cache_dir:
        os.environ["MINARI_DATASETS_PATH"] = args.cache_dir

    datasets = resolve_datasets(args)

    if args.list:
        for dataset_id in datasets:
            print(dataset_id)
        return 0

    try:
        import minari
    except ImportError as exc:
        print(
            "Minari is not installed in the current environment. "
            "Install requirements first, then rerun this script.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    total = len(datasets)
    failures = []
    for idx, dataset_id in enumerate(datasets, start=1):
        print(f"[{idx}/{total}] Downloading {dataset_id}")
        try:
            minari.download_dataset(
                dataset_id,
                force_download=args.force_download,
            )
        except Exception as exc:  # pragma: no cover - network/backend dependent
            failures.append((dataset_id, exc))
            print(f"Failed: {dataset_id}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    if failures:
        print("\nDownload finished with errors:", file=sys.stderr)
        for dataset_id, exc in failures:
            print(f"- {dataset_id}: {exc}", file=sys.stderr)
        return 1

    print(f"\nDownloaded {total} dataset(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
