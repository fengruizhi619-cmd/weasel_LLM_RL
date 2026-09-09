# -*- coding: utf-8 -*-
"""Decrypt / audit the online corpus (AES-256-GCM).

usage: python decrypt_corpus.py <corpus.jsonl> [--key PASS] [--key-file PATH]
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_server as osrv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    writer = osrv.CorpusWriter(args.path, key=args.key, key_file=args.key_file)
    records = writer.read_all()
    print("mode=%s records=%d" % ("AES-256-GCM" if writer.aead else "XOR", len(records)))
    for record in records[:args.limit or len(records)]:
        print(record)


if __name__ == "__main__":
    main()
