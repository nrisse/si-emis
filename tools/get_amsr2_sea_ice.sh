#!/bin/bash

# description
# script downloads daily sea ice data from university of bremen for given time
# just change the time range below, which is available for the campaigns
# argument order:
# first argument: output directory
# second argument: start date
# third argument: end date

# dates of campaigns:
# ACLOUD
#date_start=2017-05-19
#date_end=2017-06-27

# AFLUX
#date_start=2019-03-19
#date_end=2019-04-12

# MOSAiC-ACA
#date_start=2020-08-30
#date_end=2020-09-14

# PARMACMiP
#date_start=2018-03-23
#date_end=2018-04-05

# WALSEMA
#date_start=2022-07-01
#date_end=2022-09-01

# source
# https://seaice.uni-bremen.de/start/data-archive/
# additionally lat and lon grids have to be downloaded
# https://seaice.uni-bremen.de/data/grid_coordinates/n6250/

# get arguments
out_dir=$1
date_start=$2
date_end=$3
source='https://seaice.uni-bremen.de/data/amsr2/asi_daygrid_swath/n6250'

d=$date_start
while [ "$d" != $date_end ]; do

  echo $d

  # get day month year and month in string
  DAY=$(date -d "$d" '+%d')
  MONTH=$(date -d "$d" '+%m')
  YEAR=$(date -d "$d" '+%Y')
  MONTH_STR=$(date -d "$d" '+%b')
  MONTH_STR="${MONTH_STR,,}"

  # download data as hdf, don't forget the lat lon information
  file=$source/$YEAR/$MONTH_STR/Arctic/asi-AMSR2-n6250-$YEAR$MONTH$DAY-v5.4.hdf
  echo $file
  wget --directory-prefix $out_dir $file

  # next date
  d=$(date -I -d "$d + 1 day")

done
