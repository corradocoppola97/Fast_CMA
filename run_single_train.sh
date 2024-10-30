#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing network (network)"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing optimizer (opt)"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Missing dataset (dts)"
    exit 1
fi


docker run -e argg="1" -v ...your volumes... /usr/bin/python3 /work/project/main.py --network "${1}" --opt "${2}" --dts "${3}"
