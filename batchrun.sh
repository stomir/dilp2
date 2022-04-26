#!/bin/bash

POSITIONAL_ARGS=()
SRUN="srun -E -c 31 --gpus-per-node=4"
FROM="1"
KEEP=""
TMP=""
TIMES="1"

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--from)
      FROM="$2"
      shift
      shift
      ;;
    -S|--slurm)
      SRUN="srun $2"
      shift
      shift
      ;;
    -s|--add_slurm)
      SRUN="$SRUN $2"
      shift
      shift
      ;;
    -k|--keep)
      KEEP="yes"
      shift
      ;;
    --times)
      TIMES="$2"
      shift
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

if [ -z "$TMP" ]; then
  TMP=`mktemp -d`
fi

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

echoo () {
  echo $1 | tee -a $TMP/report
}

echoo "problem: $EXAMPLE"
echoo "seeds: $FROM - $TO"
echoo "output: $TMP"
echoo "flags: $FLAGS"
echoo "repetitions: $TIMES"

set CUBLAS_WORKSPACE_CONFIG=":4096:8"

for i in `seq -w $FROM $TO`; do
  ( 
    for t in `seq 1 $TIMES`; do
      $SRUN -J dilp/`basename $EXAMPLE`/$i/$t/`basename $TMP` python3 run.py $EXAMPLE --seed $i $FLAGS 2> >(tee -a $TMP/$i.err 1>&2) >> $TMP/$i
    done
  )&
done
wait || exit $?

results () {
  for i in `seq -w $FROM $TO`; do
    echo -n "$i: "
    cat $TMP/$i | grep result | tail -n 1
  done
}

echoo "all results:"
results | tee -a $TMP/report
OK=`results | grep "OK" | wc -l`
ALL=`results | wc -l`
FUZZY=`results | grep -e OK -e FUZZY | grep -v OVERFIT | wc -l`
TRAIN=`results | grep -e OK -e OVERFIT | wc -l`
FUZZY_OVERFIT=`results | grep -e OK -e FUZZY -e OVERFIT | wc -l`
echoo "all correct: $OK/$ALL"
echoo "fuzzily correct: $FUZZY/$ALL"
echoo "correct on training: $TRAIN/$ALL"
echoo "fuzzily correct on training: $FUZZY_OVERFIT/$ALL"
if [ -n "$KEEP" ]; then
  >&2 echo "all results in $TMP"
else 
  rm -r $TMP
fi