#!/bin/bash

POSITIONAL_ARGS=()
SRUN="srun -E -c 31"
FROM="1"

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

TMP=`mktemp -d`
for i in `seq -w $FROM $TO`; do
  ( $SRUN python3 run.py $EXAMPLE $FLAGS --seed $i > $TMP/$i )&
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