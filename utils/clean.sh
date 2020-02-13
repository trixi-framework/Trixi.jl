#!/bin/bash

# Delete *.mem files
printf "Deleting all *.mem files... "
find . -name "*.mem" -print0 | xargs -0 rm
echo "done"
