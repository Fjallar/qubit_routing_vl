#!/bin/bash

## general settings
# JOB_ID=$1
JOB_PATH=$1

## stable-baselines
ALGO='dqn2'
ENV_ID=qcirc-routing-v0
N_ENVS=32
LOG_INT=$((6*$N_ENVS))

echo python3 train.py --algo $ALGO --env $ENV_ID --tensorboard-log $JOB_PATH --gym-packages "pycirc" --vec-env "subproc" --log-interval $LOG_INT -params n_envs:$N_ENVS

# training command
if ! command -v singularity /dev/null; then
    echo "singularity could not be found, running locally."
    python train.py --algo $ALGO --env $ENV_ID \
        --tensorboard-log $JOB_PATH \
        --gym-packages "pycirc" \
        --vec-env "subproc" \
        --log-interval $LOG_INT \
        -params n_envs:$N_ENVS \
        # custom_algo:'{"label":"mdqn", "class":"sb3_contrib.dqn_mask.MaskableDQN"}' \
        # --trained-agent "rl-trained-agents/$ALGO/$ENV_ID/$MODEL"
else
    singularity exec --nv ~/quantum_circuit_optimization/rl-gym-qco.simg python3 train.py --algo $ALGO --env $ENV_ID \
        --tensorboard-log $JOB_PATH \
        --gym-packages "pycirc" \
        --vec-env "subproc" \
        --log-interval $LOG_INT \
        -params n_envs:$N_ENVS \
        # custom_algo:'{"label":"rainbow", "class":"sb3_contrib.rainbow.Rainbow"}' \
        # --trained-agent "rl-trained-agents/$ALGO/$ENV_ID/$MODEL"
fi
