


# Instructions for running this code:
- Install Cmake (latest version).
- Open `CMake`. Assuming you cloned the repo into the folder `D:\GITLAB\smiley_detection`:
  - In the line: `Where is the source code:` enter `D:\GITLAB\smiley_detection`.
  - In the line: `Where to build the binaries` enter `D:\GITLAB\smiley_detection\build`.
- On `CMake` click on `Configure`. Then choose `visual studio 16 2019` and platform `x64`.
- It might fail at the first time you press `Configure` due to incorrect Cuda version. If so, you should change `CUDA_TOOLKIT_ROOT_DIR` to the cuda path that your OpenCV works with. In my case I changed it to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`.
- Click on `Configure` again. It should succeed this time.
- When it finishes configuring, click on `Generate`.
- Open `smiley_detection\build\smiley_detection.sln` in `Visual Studio 2019`.
- Right click on `ALL_BUILD` and then click on `Rebuild`.
- Right click on `detect_smileys_in_image_cuda` and choose `Set as Startup Project`.
- Now you can run `detect_smileys_in_image_cuda` (you can change it to `Release` to make it faster).