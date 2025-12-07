@echo off
REM Compile AudioKernel.cu to PTX for CUDA acceleration
REM Requires NVIDIA CUDA Toolkit to be installed

echo Compiling AudioKernel.cu to PTX...

REM Compile for multiple GPU architectures
REM RTX 30/40 series: compute_86, sm_86 (Ampere/Ada)
REM RTX 20 series: compute_75, sm_75 (Turing)
REM Adjust based on your GPU

nvcc -ptx AudioKernel.cu -o AudioKernel.ptx ^
    --gpu-architecture=compute_75 ^
    --gpu-code=sm_75,sm_86,sm_89 ^
    -O3

if %ERRORLEVEL% EQU 0 (
    echo AudioKernel.ptx compiled successfully!
) else (
    echo Compilation failed. Make sure CUDA Toolkit is installed.
)

pause
