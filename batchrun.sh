#!/bin/bash

POSITIONAL_ARGS=()
SRUN="srun -E -c 1 --gpus-per-node=1 -p IFIall"
FROM="1"
KEEP=""
TMP=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--from)
      FROM="$2"
      shift
      shift
      ;;
    -s|--srun)
      SRUN="srun $2"
      shift
      shift
      ;;
    -k|--keep)
      KEEP="yes"
      shift
      ;;
    -od|--outdir)
      TMP="$2"
      KEEP="yes"
      mkdir -p $TMP
      shift
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [[ "$2" =~ ^[0-9]+$ ]]; then
    FROM="$1"
    TO="$2"
    EXAMPLE="$3"
    FLAGS="${@:4}"
else
    N="$1"
    let "TO=$FROM+$N-1"
    EXAMPLE="$2"
    FLAGS="${@:3}"
fi

echo "FLAGS: $FLAGS"

set CUBLAS_WORKSPACE_CONFIG=":4096:8"

if [ -z "$TMP" ]; then
  TMP=`mktemp -d`
fi
for i in `seq -w $FROM $TO`; do
  ( $SRUN -E -J dilp/$EXAMPLE/$i/`basename $TMP` python3 run.py $EXAMPLE $FLAGS --seed $i > $TMP/$i )&
done
wait || exit $?
echo "all results:"
for i in `seq -w $FROM $TO`; do
  echo -n "$i: "
  cat $TMP/$i | grep result
done
OK=`cat $TMP/* | grep "result" | grep "OK" | wc -l`
ALL=`cat $TMP/* | grep "result" | wc -l`
FUZZY=`cat $TMP/* | grep "result" | grep -v fuzzily_valid_worlds=0 | wc -l`
echo "final: $OK/$ALL"
echo "fuzzily correct: $FUZZY/$ALL"
if [ -n "$KEEP" ]; then
  >&2 echo "all results in $TMP"
else 
  rm -r $TMP
fi