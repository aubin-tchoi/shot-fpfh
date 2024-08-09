# 🌐 SHOT / 📊 FPFH descriptors on 3D Point Clouds

This repository contains resources to compute SHOT and FPFH, two 3D descriptors on point clouds.
The main script `register_point_clouds.py` contains a pipeline consisting of the following steps:

- **Data retrieving**: binary `.ply` files are supported. Two point clouds will be loaded.
- **Query point selection**: an algorithm taken from `query_points_selection.py` will be used to select a subset of
  query points.
- **Descriptors computation**: descriptors will be computed on the query points.
- **Descriptors matching**: descriptors between the two point clouds will be matched using one of the algorithms found
  in `matching.py` and a coarse registration will be performed.
- **Fine registration**: an ICP-based registration will be performed.
- **Data saving**: if not disabled, 4 point clouds will be saved: 2 for each descriptor, one for the coarse registration
  and another one for the fine registration.

## 💬 Basic usage

Install the project using `poetry install` and run `poetry run register_point_clouds --help`
to display a help message specifying the arguments of this main script.
This script will run the steps specified above in the same order, each argument finding a role in one of the steps.

## 📦 Dependency management

Dependencies are managed using poetry: https://python-poetry.org/

## 🔧 Implementation

The descriptors are implemented from scratch based on `NumPy`. The `KDTree` implementation used here is the one from
`scikit-learn.neighbors`.

### Possible performance-improving extensions

- Using `numba` to jit the code. The compilation is quite long and the jitting fails on various functions
  like `KDTree.query_radius` or `np.mean` (https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean). It
  could work with some additional work to refactor the code in a `numba`-friendly manner.

- Use the `KDTree` from `scipy` instead of `scikit-learn` to allow for parallel computing on several workers at once.
