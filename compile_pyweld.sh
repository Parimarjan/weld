#!/bin/bash
cargo check
cargo build --release
cd ~/weld/python/pyweld
python setup.py install --user
cd -
cd ~/weld/python/numpy
pytest tests-gpu.py
cd -
