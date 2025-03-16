from spatialdata_io import merscope
import spatialdata as sd

path = './HumanBreastCancerPatient1/'
path_read = path + "data"
path_write = path + "data.zarr"


print("parsing the data... ", end="")
sdata = merscope(path)
print("done")

print("writing the data... ", end="")
sdata.write(path_write)
print("done")

sdata = sd.SpatialData.read(path+"data.zarr/")
print(sdata)
