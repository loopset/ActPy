#!/bin/sh

# Self locate script when sourced
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# And now export
export PYTHONPATH=${PYTHONPATH}:"${SCRIPT_DIR}/src/"
