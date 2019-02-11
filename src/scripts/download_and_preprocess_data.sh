#!/usr/bin/env bash
export PYTHONPATH=..
python scripts/kaggle_data_download.py
python scripts/external_data_download.py
python scripts/image_merging.py
python scripts/data_deduplication.py