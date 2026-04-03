#!/bin/bash
#
# TOSCA Animal → Shape Indices Map
#
# Edit the index lists below to control which shapes are processed by any
# script that reads --animals.  Simply remove an index from the list to
# exclude that shape from all downstream steps (render, train, geodesic …).
#
# Format:  ANIMAL_INDEX_MAP["<animal>"]="<idx0> <idx1> ..."
# Gaps in the numeric sequence are intentional (some TOSCA meshes are missing).
#
# Source this file from any script that needs the map:
#   source "$(dirname "${BASH_SOURCE[0]}")/tosca_animal_map.sh"
# Then call expand_animals "<comma-list>" to get a comma-separated shape list.
# ============================================================================

declare -A ANIMAL_INDEX_MAP

ANIMAL_INDEX_MAP["cat"]="0 1 2 3 4 5" #  7 8 10"
ANIMAL_INDEX_MAP["centaur"]="0 1 2 3 4 5" # "
ANIMAL_INDEX_MAP["david"]="0 1 2 3 4 5" # 6 7 8 9 10 11 12 13 14"
ANIMAL_INDEX_MAP["dog"]="0 1 2 3 4 5" #  6 7 8 9 10"
ANIMAL_INDEX_MAP["gorilla"]="0 1 2 3 4 5" # 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
ANIMAL_INDEX_MAP["horse"]="0 2 3 4 5 6" # 7 8 9 10 12 13 14 15 16 17 18"
ANIMAL_INDEX_MAP["lioness"]="0 1 2 3 4 5" #  6 7 8 10 12 13 14 15 16"
ANIMAL_INDEX_MAP["michael"]="0 1 2 3 4 5" # 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
ANIMAL_INDEX_MAP["seahorse"]="0 1 2 3 4 5"
ANIMAL_INDEX_MAP["shark"]="0"
ANIMAL_INDEX_MAP["victoria"]="0 1 2 3 4 5" # 6 7 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25"
ANIMAL_INDEX_MAP["wolf"]="0 1 2"

# ---------------------------------------------------------------------------
# expand_animals <comma-separated-animal-names>
#   Prints a comma-separated list of shape names (e.g. "cat0,cat1,cat2,dog0")
#   by looking up each animal in ANIMAL_INDEX_MAP.
#   Unknown animals trigger a warning but do not abort.
# ---------------------------------------------------------------------------
expand_animals() {
    local input="$1"
    local result=""
    local animals
    # Split input on commas
    IFS=',' read -ra animals <<< "$input"
    for animal in "${animals[@]}"; do
        animal="$(echo "$animal" | tr -d '[:space:]')"
        if [[ -z "${ANIMAL_INDEX_MAP[$animal]+_}" ]]; then
            echo "Warning: Animal '$animal' not found in ANIMAL_INDEX_MAP (tosca_animal_map.sh)" >&2
            continue
        fi
        # Split index list on spaces (default IFS)
        local idx
        for idx in ${ANIMAL_INDEX_MAP[$animal]}; do
            result="${result}${result:+,}${animal}${idx}"
        done
    done
    echo "$result"
}
