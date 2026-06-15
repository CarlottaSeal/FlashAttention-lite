@echo off
setlocal
cd /d %~dp0

REM Blackwell (RTX 5070 Ti, sm_120) build. Paths updated for CUDA 13.2 + MSVC 14.44 + SDK 26100.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "SDK_VER=10.0.26100.0"
set "PATH=%CUDA_PATH%\bin;%MSVC_ROOT%\bin\Hostx64\x64;%SDK_ROOT%\bin\%SDK_VER%\x64;%PATH%"
set "INCLUDE=%MSVC_ROOT%\include;%SDK_ROOT%\Include\%SDK_VER%\ucrt;%SDK_ROOT%\Include\%SDK_VER%\um;%SDK_ROOT%\Include\%SDK_VER%\shared"
set "LIB=%MSVC_ROOT%\lib\x64;%SDK_ROOT%\Lib\%SDK_VER%\ucrt\x64;%SDK_ROOT%\Lib\%SDK_VER%\um\x64"

if not exist build\bin mkdir build\bin

"%CUDA_PATH%\bin\nvcc.exe" -arch=sm_120 -O3 -std=c++17 --allow-unsupported-compiler ^
    -I src ^
    src\main.cu -o build\bin\attn.exe
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo BUILD_OK
