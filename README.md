# TET Repo1

# To use the ROKID AIR Pro
https://github.com/osztobanyipeter1/AR-Driver

## Description
To get the best surface real-time or near real-time, we can use some basic algorith to do that.
### Alpha shape
### Ball pivoting
### Poisson
For the ball pivoting and poisson method have the need to calculate the normal vectors, what takes lots of time. So I think we can skip this part and we can also skip ball pivoting and poission. The alpha shape gives back quite good surfaces real time.
The FPSample can help to reduce the size of the point cloud and make it even more faster.

## 1.ipynb: This is the file i will update with the best solution

## Other ipynb and py files are just for testing.


## Already tried algorithms:
### Point2mesh
https://github.com/ranahanocka/Point2Mesh/
Its running on linux.
Mine: https://github.com/osztobanyipeter1/point2mesh

## Shape As Points:
https://github.com/autonomousvision/shape_as_points
Not known the working requirements.
Not working.

## SLIDE:
-Requirements done
conda env create -f environment.yml
conda activate slide
cd pointnet2_ops_lib
pip install -e . <---- Done until this part
cd ..
pip install -e .

https://github.com/SLIDE-3D/SLIDE
Dont have enough space to test it.

## GeoUDF:
https://github.com/rsy6318/GeoUDF
Builded, but not running yet
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md to help me with pytorch
