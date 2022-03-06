#!/bin/bash
EXAMPLE=$3
TO=$2
FROM=$1
FLAGS="${@:4}"

RUN="srun -E -c 31"

set CUBLAS_WORKSPACE_CONFIG=":4096:8"

TMP=`mktemp -d`
for i in `seq -w $FROM $TO`; do
  ( $RUN python3 run.py $EXAMPLE $FLAGS --seed $i > $TMP/$i )&
done
wait || exit $?
echo "all results:"
for i in `seq -w $FROM $TO`; do
  echo -n "$i: "
  cat $TMP/$i | grep result
done
OK=`cat $TMP/* | grep "result" | grep "OK" | wc -l`
ALL=`cat $TMP/* | grep "result" | wc -l`
echo "final: $OK/$ALL"
rm -r $TMP
