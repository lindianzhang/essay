$vcvars = "D:\System\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

if (!(Test-Path $vcvars)) {
    throw "Visual Studio C++ Build Tools not found"
}

if (!(Test-Path "$cuda\bin\nvcc.exe")) {
    throw "CUDA Toolkit 12.4 not found"
}

# Load MSVC and Windows SDK environment
$lines = cmd.exe /c 'call "D:\System\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && set'

if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize MSVC"
}

foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable(
            $matches[1], $matches[2], "Process"
        )
    }
}

# CUDA environment
$env:CUDA_HOME = $cuda
$env:CUDA_PATH = $cuda
$env:Path = "$cuda\bin;$env:Path"

# RTX 4060 compilation settings
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:MAX_JOBS = "2"
$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"

# Real3D-AD dataset
$env:REAL3D_PCD_ROOT = "D:\Desktop\Real3D-AD-PCD"

if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe is not available"
}

Write-Host "MiniShift RTX 4060 environment ready" -ForegroundColor Green
