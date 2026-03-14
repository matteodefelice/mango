# %%
import src.mango.io.data as md 

token = "edh_pat_28142c8f3482fac46fabc58594bca5ace512a01ddceeca0799aaf19c264f80601547ea019347ac9b0b4c2f10dcb6da77"
df1 = md.data_cube_access(method='edh', token=token)
df1.select_data(
    lat_range=(44, 36), 
    lon_range=(360 - 97, 360 - 83), 
    time_range=('2014', '2024'))
df1.get_daily()
# print(df1.data_daily)

# df2 = data_cube_access(method='earthmover', token=token, 
                    # lat_range=(39, 35), lon_range=(11, 16), time_range=('2023', '2025'))
# print(df2.data_daily)