#
# !!!Area: North/West/South/East!!!
#

import cdsapi
import datetime

c = cdsapi.Client()

for i in range(1979,2021):
    int_time_obj = datetime.datetime.strptime(str(i)+'0101', '%Y%m%d')
    end_time_obj = datetime.datetime.strptime(str(i)+'0101', '%Y%m%d')
    day_delta=datetime.timedelta(days=1)
    curr_time_obj = int_time_obj

    # mon looper
    for j in range(0,12):
        
        curr_year=curr_time_obj.strftime('%Y')
        curr_mon=curr_time_obj.strftime('%m')
        dayinmon_list=[]

        # obtain day list
        while curr_time_obj.strftime('%m') ==curr_mon:
            dayinmon_list.append(curr_time_obj.strftime('%d'))
            curr_time_obj=curr_time_obj+day_delta
       
        # retrieve data
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type':'reanalysis',
                'format':'netcdf',
                'variable':[
                    '10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure',
                ],
                'year':curr_year,
                'month':curr_mon,
                'day':dayinmon_list,
                'area':'60/60/-10/170',
                'time':'00/to/23/by/6',
            },
            curr_year+curr_mon+'-surf.nc')
