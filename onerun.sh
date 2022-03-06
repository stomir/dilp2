#!/bin/bash
#RUN='srun -E'
RUN='srun -p IFItitan -c 31 -E'
EXAMPLE=$1
FLAGS="${@:2}"

TMP=`mktemp`
$RUN python3 sample.py $EXAMPLE $FLAGS > $TMP
