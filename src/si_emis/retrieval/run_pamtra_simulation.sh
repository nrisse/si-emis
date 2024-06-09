#!/bin/bash

# Description
# Runs PAMTRA simulation for a specific research flight. Either the aircraft
# geometry is simulated as a function of aircraft time, or the satellite
# geometry is simulated as a function of satellite footprint index that aligns
# with the aircraft for a given research flight.
#
# How to edit the script to run various flights:
# - provide list of flight_ids
# - optional: provide list of setup_names
# - optional: specify that all setup names should be simulated (default)
# then run this bash script with an argument: "aircraft" or "satellite"
#
# Keep track of simulations in table in case only some settings are updated

# for aircraft
flight_ids=(
  "ACLOUD_P5_RF23"
  "ACLOUD_P5_RF25"
  "AFLUX_P5_RF08"
  "AFLUX_P5_RF14"
  "AFLUX_P5_RF15"
)

# for satellite
#flight_ids=(
#"ACLOUD_P5_RF23"
#"ACLOUD_P5_RF25"
#"AFLUX_P5_RF08"
#"AFLUX_P5_RF14"
#"AFLUX_P5_RF15"
#  )

setup_names="" # runs all setups

# for aircraft special
#setup_names=("09" "10" "11" "12" "13" "14" "15" )  # Lamb, e=1
#setup_names=("02" "09")  # Lamb (e=1, e=0), no sensitivity tests

# for satellite special
#setup_names=("00" "01")  # only natural
#setup_names=("03" "05")  # only for e=1
#setup_names=("02" "03" "04" "05")  # only idealized

all_setup_names=true

for flight_id in ${flight_ids[*]}; do
  echo "Time: $(date)" >>run_pamtra_simulation.log
  python -m retrieval.pamsim "$flight_id" "$1" \
    --setup_names "${setup_names[@]}" \
    --all_setup_names $all_setup_names >>run_pamtra_simulation.log
done
