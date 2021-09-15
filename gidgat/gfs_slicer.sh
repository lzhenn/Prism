#!/bin/sh
#------------------------------------------------------
# GFS Slicer using NOAA grib filter to extract
# orginal GFS 0d25 products within domains of interests 
# and variables for driving PATH system. 95% space can 
# be saved compared with original global data.
#
#                                    Zhenning LI
#                                   July 1, 2021
#------------------------------------------------------

#------------------------------------------------------
# USAGE: 
# Plese first assign your subset domain (LAT_BOTTOM,
#   LAT_TOP, LON_LEFT, LON_RIFHT) below
# 
# If use from external call:
#   sh gfs_slicer.sh $ARCH_PATH $INIT_YYYYMMDDHH $FCST_DAYS
#   e.g. sh gfs_slicer.sh /home/metctm1/array/data 2021070512 1
#    
# Or you could remove the comment # from defined variables and
# assign them properly, just type: sh gfs_slicer.sh
#------------------------------------------------------



# ------------Below for user-defined configurations ------------
# Archive path
ARCH_PATH=$1
#ARCH_PATH=/home/lzhenn/array74/Njord_Calypso/drv_field/gfs/2021062712

# Start time 
STRT_YMDH=$2
#STRT_YMDH=2021062712

# How long period to fecth
FCST_DAY=$3
#FCST_DAY=1

LON_LEFT=$4
LON_RIGHT=$5
LAT_TOP=$6
LAT_BOTTOM=$7

# The interval to fetch GFS output, 3-hr preferred, 
# 1-hr minimum, and no longer than 6-hr.
FRQ=$8


# ------------Upper for user-defined configurations ------------

TOTAL_HR=`expr $FCST_DAY \* 24`

BASE_URL="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


LV_FILTER="&lev_500_mb=on&lev_10_m_above_ground=on&lev_mean_sea_level=on"
VAR_FILTER="&var_HGT=on&var_PRMSL=on&var_UGRD=on&var_VGRD=on"
DOMAIN_FILTER="&subregion=&leftlon="${LON_LEFT}"&rightlon="${LON_RIGHT}"&toplat="${LAT_TOP}"&bottomlat="${LAT_BOTTOM}
TS_FILTER="&dir=%2Fgfs."${STRT_YMDH:0:8}"%2F"${STRT_YMDH:8}"%2Fatmos"

cd ${ARCH_PATH}
rm -f gfs.t*grib2

for CURR_HR in $(seq 0 $FRQ $TOTAL_HR) 
do
    
    TSTEP=`printf "%03d" $CURR_HR`
    
    FN_FILTER="?file=gfs.t"${STRT_YMDH:8}"z.pgrb2.0p25.f"${TSTEP}
    
    SRC_URL=${BASE_URL}${FN_FILTER}${LV_FILTER}${VAR_FILTER}${DOMAIN_FILTER}${TS_FILTER}

    wget ${SRC_URL} -O ${ARCH_PATH}/${FN_FILTER:6}".grib2"
    echo "convert "${FN_FILTER:6}".grib2 to netCDF"

    ncl_convert2nc ${FN_FILTER:6}.grib2
done 
