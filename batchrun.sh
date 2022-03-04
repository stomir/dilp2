#!/bin/bash
EXAMPLE=$1
RUNS=$2
FLAGS="${@:3}"

set CUBLAS_WORKSPACE_CONFIG=":4096:8"

TMP=`mktemp -d`
for i in `seq 1 $RUNS`; do
  ( bash onerun.sh $EXAMPLE $FLAGS --seed $i > $TMP/$i )&
done
wait || exit $?
cat $TMP/*
rm -r $TMP
