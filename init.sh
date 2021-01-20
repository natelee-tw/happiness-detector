#!/bin/sh

if [[ -z "$1" ]]
then
    echo "Cleaning up"
    rm init.sh
    rm ./.polyaxon/.polyaxonproject
    rm -rf .git
    read -p "Enter name for your project: " projectName
    echo "Renaming folder to $projectName"
    mv "$PWD" "${PWD%/*}/$projectName"
    cd -P .
else
    echo "Test run. Skipping clean up."
fi
