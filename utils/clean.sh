#!/bin/bash

# Delete *.mem files
printf "Deleting all *.mem files... "
find . -name "*.mem" -print0 | xargs -0 rm -f
echo "done"

# Delete *.cov files
printf "Deleting all *.cov files... "
find . -name "*.cov" -print0 | xargs -0 rm -f
echo "done"
