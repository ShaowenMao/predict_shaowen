#!/usr/bin/env python3
"""Extract one verified 522-row sampling table for each qualification case."""

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-csv", required=True, type=Path)
    parser.add_argument("--cases-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.cases_csv.open(newline="", encoding="utf-8") as stream:
        case_rows = list(csv.DictReader(stream))
    wanted = {(row["geology_id"], row["case_id"]) for row in case_rows}
    if len(wanted) != 24:
        raise ValueError("Qualification case list must contain 24 unique cases")

    streams = {}
    writers = {}
    counts = {key: 0 for key in wanted}
    paths = {}
    try:
        with args.sampling_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError("Sampling CSV has no header")
            for key in sorted(wanted):
                geology_id, case_id = key
                path = args.output_root / (
                    "{}_case{:02d}_slice_window_values.csv".format(
                        geology_id, int(case_id)
                    )
                )
                stream = path.open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                writer.writeheader()
                streams[key] = stream
                writers[key] = writer
                paths[key] = path

            for row in reader:
                key = (row["geology_id"], row["case_id"])
                if key in wanted:
                    writers[key].writerow(row)
                    counts[key] += 1
    finally:
        for stream in streams.values():
            stream.close()

    bad = {key: count for key, count in counts.items() if count != 522}
    if bad:
        raise ValueError("Qualification case row-count mismatch: {}".format(bad))

    summary_path = args.output_root / "qualification_input_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["geology_id", "case_id", "row_count", "file_name", "bytes", "sha256"]
        )
        for key in sorted(wanted):
            path = paths[key]
            writer.writerow(
                [key[0], key[1], counts[key], path.name, path.stat().st_size, sha256_file(path)]
            )
    print("Prepared {} qualification inputs with {} total rows".format(
        len(wanted), sum(counts.values())
    ))
    print(summary_path)


if __name__ == "__main__":
    main()
