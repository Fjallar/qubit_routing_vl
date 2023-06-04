#!/bin/bash

# Path to the root directory of your main repository
root_dir="."

# Create a symlink for the file
ln -s "$root_dir/pycirc.py" "$root_dir/rl-baselines3-zoo/pycirc.py"

# Create a symlink for the folder
ln -s "$root_dir/dqn2" "$root_dir/rl-baselines3-zoo/dqn2"
