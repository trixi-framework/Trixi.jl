#!/bin/bash

if [ $# -lt 1 ]; then
  echo "missing argument: output filename" >&2
  exit 2
fi

output_filename="$1"

ffmpeg -framerate 20 -f image2 -i %*.png -pix_fmt yuv420p "$output_filename"
