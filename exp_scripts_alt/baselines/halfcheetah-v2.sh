#!/bin/bash

if [ -z "$3" ]; then
    echo "target 'local' selected automatically."
    target="local"
    tag=$1
    reps=$2
else
   target=$1
   tag=$2
   reps=$3
fi 


seed1=`od -A n -t d -N 2 /dev/urandom`
seed2=`od -A n -t d -N 2 /dev/urandom`
seed3=`od -A n -t d -N 2 /dev/urandom`
cmd_line=" --model ICNN --env HalfCheetah-v2 --outdir /icnn/results/${tag} --total 100000 --force --train 100 --test 1 --tfseed ${seed1} --npseed ${seed2} --gymseed ${seed3} "

${ICNN_PATH}/exp_scripts_alt/run.sh "${target}" "${cmd_line}" "${tag}" "${reps}"
