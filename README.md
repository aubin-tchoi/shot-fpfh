# SHOT / FPFH descriptors on 3D Point Clouds

## Things that did not work but maybe could work

Using `numba` to jit the code. The compilation is quite long and the jitting fails on various functions
like `KDTree.query_radius` or `np.mean` (https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean).
It could work with some additional work to refactor the code in a `numba`-friendly manner.
