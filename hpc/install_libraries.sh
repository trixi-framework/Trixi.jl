#!/usr/bin/env bash
set -euo pipefail

: "${EBROOTOPENMPI:?Source hpc/env.sh first.}"
: "${EBROOTZLIB:?Source hpc/env.sh first.}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
native_prefix=/scratch/hpc-prf-cobentx/depot/native/otus-foss-2025a-v2
# CMake uses its embedded library paths because its module overrides Julia's libcurl.
cmake=/opt/software/pc2/EB-SW/software/CMake/3.31.3-GCCcore-14.2.0/bin/cmake
fingerprint=$({
    sha256sum < "$root/hpc/libraries/CMakeLists.txt"
    printf '%s\n' "$EBROOTOPENMPI" "$EBROOTZLIB" "$cmake"
} | sha256sum)
fingerprint=${fingerprint%% *}
if [[ ${1:-} == --print-fingerprint ]]; then
    printf '%s\n' "$fingerprint"
    exit 0
fi
[[ $# == 0 ]] || { echo 'Expected no arguments or --print-fingerprint.' >&2; exit 1; }
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
mkdir -p -- "$(dirname -- "$native_prefix")"
exec 9>"$native_prefix.lock"
flock 9

libraries=(p4est/lib/libp4est.so p4est/lib/libsc.so
           t8code/lib/libt8.so t8code/lib/libp4est.so t8code/lib/libsc.so)
if [[ -e "$native_prefix" ]]; then
    if [[ ! -f "$native_prefix/VERSION" ||
          $(cat "$native_prefix/VERSION") != "$fingerprint" ]]; then
        echo "Native installation at $native_prefix does not match the build recipe and toolchain." >&2
        echo 'Select a new versioned path in the installer and preferences.' >&2
        exit 1
    fi
    for library in "${libraries[@]}"; do
        test -s "$native_prefix/$library"
    done
    exit 0
fi

stage=$(mktemp -d "$native_prefix.build.XXXXXX")
installed=false
cleanup() {
    if [[ $installed == false && -d "$native_prefix" ]]; then
        [[ $(realpath -- "$native_prefix") == "$native_prefix" ]] || return 1
        rm -r -- "$native_prefix"
    fi
    case "$(realpath -- "$stage")" in
        "$native_prefix".build.*) rm -r -- "$stage" ;;
        *) echo 'Refusing to clean an unexpected build directory.' >&2 ;;
    esac
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$native_prefix"

"$cmake" -S "$root/hpc/libraries" -B "$stage" \
    -G 'Unix Makefiles' -DCMAKE_INSTALL_PREFIX="$native_prefix"
# Each dependency uses CMAKE_BUILD_PARALLEL_LEVEL without overlapping another build.
"$cmake" --build "$stage" --parallel 1
for library in "${libraries[@]}"; do
    test -s "$native_prefix/$library"
done
printf '%s\n' "$fingerprint" > "$stage/VERSION"
mv -- "$stage/VERSION" "$native_prefix/VERSION"
installed=true
