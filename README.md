This frankenstein repo contains the work I have done so far on the CV portion of a Connected and Automated Vehicle (CAV) simulator research project.  

There is a requirements.txt which can be installed with  
```
python install -r requirements.txt
```
However as of right now it is not updated. You would need to install nerfstudio and ultralytics to use all of the files.

Worked on:
  - Build 3d spherical view from panoramic images and MiDaS depth estimation
  - Creating a realistic image layer on top of driving simulation using NeRFs from sparse imagery
  - Recreating accurate road geometry starting from GSV -> segment road features + backproject -> fit RANSAC road plane to concatenated 3d points

Current Objective:
  - Make a dashcam pipeline that uses YoLo to segment driving related features (pedestrians, cars, traffic lights, etc.) and monocular depth  
    estimation to create digital twins in "real-time" ... Kinda like the stuff that you see on the screen on a tesla
