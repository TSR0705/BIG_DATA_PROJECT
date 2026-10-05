# Dot-source at the start of every work session:   . .\scripts\env.ps1
$env:JAVA_HOME = "C:\Java\jdk-17.0.20.1+1"
$env:HADOOP_HOME = "C:\hadoop"
$env:Path = "C:\Java\jdk-17.0.20.1+1\bin;C:\hadoop\bin;" + $env:Path
$env:PYSPARK_PYTHON = "$PSScriptRoot\..\venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "$PSScriptRoot\..\venv\Scripts\python.exe"
. "$PSScriptRoot\..\venv\Scripts\Activate.ps1"
Write-Host "BIG_DATA_PROJECT environment active (JDK 17, Hadoop shim, venv)" -ForegroundColor Green
