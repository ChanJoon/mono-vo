# Monocular visual odometry

### Prerequisite

This repository's code is specifically tailored to work with the KITTI Visual Odometry Dataset. Users interested in utilizing or experimenting with this code will need to download [KITTI Visual Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### How to use

```bash
mkdir build && cd build
cmake ..
cmake --build .
./MonoVO <sequence> # 00 to 10
```

### TODO
- [ ] : Calculate relative scale and update Coordinates
- [ ] : Refactoring

### References
- [Tutorial on Visual Odometry by D.Scaramuzza](https://rpg.ifi.uzh.ch/visual_odometry_tutorial.html)
- [CSC2541 (2016), Patrick McGarey](https://www.cs.toronto.edu/~urtasun/courses/CSC2541/03_odometry.pdf)
- [Visual Odometry Features, Tracking, Essential Matrix, and RANSAC, Stephan Weiss](https://www.cs.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A11-VisualOdometry.pdf)
- [Avi Singh's blog](https://avisingh599.github.io/vision/monocular-vo/)