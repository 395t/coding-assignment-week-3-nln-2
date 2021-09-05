#! /bin/bash
mkdir logs
pip install imageio
pip install tensorboard
python3 Setup/stl10_input.py
python3 Setup/stl10_valid.py
