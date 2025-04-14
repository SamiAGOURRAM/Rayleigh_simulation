#!/bin/bash
# run.sh

# Default parameters
NX=1024
NY=512
STEPS=1000
DUMP=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -nx)
        NX="$2"
        shift
        shift
        ;;
        -ny)
        NY="$2"
        shift
        shift
        ;;
        -steps)
        STEPS="$2"
        shift
        shift
        ;;
        -dump)
        DUMP="$2"
        shift
        shift
        ;;
        *)
        shift
        ;;
    esac
done

echo "Running Rayleigh-Taylor simulation with grid $NX x $NY for $STEPS steps"
./rt -nx $NX -ny $NY -steps $STEPS -dump $DUMP