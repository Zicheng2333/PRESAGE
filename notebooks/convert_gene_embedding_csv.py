#!/usr/bin/env python3
"""Convert gene-embedding CSV (topics x genes) into PRESAGE .pkl format.

python convert_gene_embedding_csv.py \
  --input /raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/pca_embed.csv \
  --output /raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/pca_embed.pkl


python convert_gene_embedding_csv.py \
  --input /raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/state_embed.csv \
  --output /raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/state_embed.pkl

Input CSV format (example):
    ,NOC2L,KLHL17,HES4,...
    topic_0,9.03e-05,1.12e-04,2.54e-03,...
    topic_1,6.56e-05,1.51e-05,7.35e-06,...

Output format:
    A pandas DataFrame pickled with:
      - index = gene symbols
      - columns = embedding dimensions (topic_0..topic_N)
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert topic-by-gene CSV to PRESAGE embedding pickle."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file (topics x genes).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .pkl file (genes x embedding dims).",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="CSV separator (default: ,).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, sep=args.sep, index_col=0)
    if df.empty:
        raise ValueError(f"Input CSV has no data: {input_path}")

    embedding_df = df.transpose()
    embedding_df.index.name = "gene"

    if embedding_df.index.has_duplicates:
        dupes = embedding_df.index[embedding_df.index.duplicated()].unique().tolist()
        raise ValueError(
            "Duplicate gene symbols found after transpose. "
            f"Please deduplicate first. Duplicates: {dupes[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_pickle(output_path)

    print(f"Saved {embedding_df.shape} embedding matrix to {output_path}")


if __name__ == "__main__":
    main()
