ARGS="$@"

tab () {
    LINE=`cat $1/report | grep -e "^$2:" -e ": $2:" | head -n 1`
    echo -e -n "$LINE\t"
}

cat $1/report | head -n 4

for arg in `echo $@ | /usr/bin/python3 -m natsort`; do
    LINE=`basename $arg`
    echo -e -n "$LINE\t"
    tab $arg "all correct"
    tab $arg "fuzzily correct"
    tab $arg "correct on training"
    tab $arg "fuzzily correct on training"
    echo ""
done
