<#
  setup_phase0.ps1  -  BIG_DATA_PROJECT  Phase 0: Environment Setup (Windows)

  RUN (normal PowerShell window, not admin - UAC prompts will appear for installers):
      cd C:\Users\ACER\Desktop\BIG_DATA_PROJECT
      powershell -ExecutionPolicy Bypass -File .\scripts\setup_phase0.ps1

  What it does (re-runnable / idempotent):
    1. Java 17 (Temurin)  via winget  - project-scoped, does NOT overwrite an existing global JAVA_HOME
    2. Python 3.11        via winget
    3. winutils + hadoop.dll (Hadoop 3.3.6) into C:\hadoop\bin  - required by Spark on Windows
    4. Project folder scaffold + .gitignore hardening
    5. venv + pip install (requirements.txt) + requirements.lock.txt
    6. scripts\verify_env.py  (Spark groupBy/shuffle/Parquet round-trip + ML stack imports)
    7. Git: branch feature/phase0-environment, commit, tag v0.1-environment (only if verified)

  Logs:  logs\phase0_setup.log   logs\phase0_verify.log   logs\phase0_verify.ok (success marker)
  Flags: -SkipInstalls  (skip winget steps if Java 17 / Python 3.11 are already installed)
#>
param([switch]$SkipInstalls)

$ErrorActionPreference = "Continue"
$Proj = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Proj
New-Item -ItemType Directory -Force -Path "$Proj\logs" | Out-Null
Start-Transcript -Path "$Proj\logs\phase0_setup.log" -Force | Out-Null

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
$summary = [ordered]@{}

# ---------------------------------------------------------------- 1. winget
Step "1/8  Checking winget"
$haveWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)
Write-Host "winget available: $haveWinget"
if (-not $haveWinget -and -not $SkipInstalls) {
    Write-Warning "winget not found. Install 'App Installer' from Microsoft Store, OR install Temurin JDK 17 + Python 3.11 manually and re-run with -SkipInstalls."
}

# ---------------------------------------------------------------- 2. Java 17
Step "2/8  Java 17 (Eclipse Temurin)"
function Find-Jdk17 {
    $roots = @("C:\Program Files\Eclipse Adoptium", "C:\Program Files\Java", "$env:LOCALAPPDATA\Programs\Eclipse Adoptium")
    foreach ($r in $roots) {
        if (Test-Path $r) {
            $d = Get-ChildItem $r -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -like "jdk-17*" } | Sort-Object Name -Descending | Select-Object -First 1
            if ($d -and (Test-Path "$($d.FullName)\bin\java.exe")) { return $d.FullName }
        }
    }
    return $null
}
$jdk = Find-Jdk17
if (-not $jdk -and -not $SkipInstalls -and $haveWinget) {
    Write-Host "Installing Temurin JDK 17 via winget (accept the UAC prompt)..."
    winget install --id EclipseAdoptium.Temurin.17.JDK -e --silent --accept-package-agreements --accept-source-agreements
    $jdk = Find-Jdk17
}
if ($jdk) {
    Write-Host "JDK 17 found: $jdk"
    $env:JAVA_HOME = $jdk
    $env:Path = "$jdk\bin;" + $env:Path
    $globalJava = [Environment]::GetEnvironmentVariable("JAVA_HOME", "User")
    if (-not $globalJava) {
        [Environment]::SetEnvironmentVariable("JAVA_HOME", $jdk, "User")
        Write-Host "User JAVA_HOME was empty -> set to JDK 17."
    } else {
        Write-Host "Existing user JAVA_HOME left untouched ($globalJava); this project pins JDK 17 via configs\spark.yaml."
    }
    & "$jdk\bin\java.exe" -version 2>&1 | ForEach-Object { Write-Host "  $_" }
    $summary["Java 17"] = "OK  ($jdk)"
} else {
    Write-Warning "JDK 17 not found. Install Temurin 17 from https://adoptium.net/temurin/releases/?version=17 then re-run."
    $summary["Java 17"] = "MISSING"
}

# ---------------------------------------------------------------- 3. Python 3.11
Step "3/8  Python 3.11"
function Find-Py311 {
    try { $p = & py -3.11 -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "C:\Python311\python.exe", "C:\Program Files\Python311\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}
$py311 = Find-Py311
if (-not $py311 -and -not $SkipInstalls -and $haveWinget) {
    Write-Host "Installing Python 3.11 via winget..."
    winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
    $py311 = Find-Py311
}
if (-not $py311) {
    Write-Error "Python 3.11 not available. Install from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and re-run."
    $summary["Python 3.11"] = "MISSING"
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "Python 3.11: $py311"
& $py311 --version
$summary["Python 3.11"] = "OK  ($py311)"

# ---------------------------------------------------------------- 4. winutils
Step "4/8  winutils + hadoop.dll (Hadoop 3.3.6) for Spark on Windows"
$HadoopHome = "C:\hadoop"
New-Item -ItemType Directory -Force -Path "$HadoopHome\bin" | Out-Null
$base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
$wuOK = $true
foreach ($f in @("winutils.exe", "hadoop.dll")) {
    $dest = "$HadoopHome\bin\$f"
    if (-not (Test-Path $dest) -or (Get-Item $dest).Length -lt 10000) {
        Write-Host "Downloading $f ..."
        try { Invoke-WebRequest -Uri "$base/$f" -OutFile $dest -UseBasicParsing }
        catch { Write-Warning "Download failed for $f : $($_.Exception.Message)"; $wuOK = $false }
    }
    if ((Test-Path $dest) -and (Get-Item $dest).Length -ge 10000) { Write-Host "  $f  OK  ($((Get-Item $dest).Length) bytes)" }
    else { Write-Warning "  $f missing or too small - download manually from $base/$f into $HadoopHome\bin"; $wuOK = $false }
}
[Environment]::SetEnvironmentVariable("HADOOP_HOME", $HadoopHome, "User")
$env:HADOOP_HOME = $HadoopHome
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$HadoopHome\bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$HadoopHome\bin", "User")
}
$env:Path = "$HadoopHome\bin;" + $env:Path
$summary["winutils"] = $(if ($wuOK) { "OK  ($HadoopHome\bin)" } else { "CHECK" })

# ---------------------------------------------------------------- 5. scaffold + .gitignore
Step "5/8  Project scaffold"
$dirs = @("data\bronze", "data\silver", "data\gold", "configs", "notebooks", "models", "reports\figures",
          "reports\profiling", "dashboard\pages", "tests", "scripts", "logs",
          "src\common", "src\ingestion", "src\preprocessing", "src\features", "src\training",
          "src\evaluation", "src\explainability", "src\inference")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $Proj $d) | Out-Null }
foreach ($pkg in @("src", "src\common", "src\ingestion", "src\preprocessing", "src\features", "src\training",
                   "src\evaluation", "src\explainability", "src\inference", "tests")) {
    $init = Join-Path $Proj "$pkg\__init__.py"
    if (-not (Test-Path $init)) { New-Item -ItemType File -Path $init | Out-Null }
}
# harden .gitignore (append only what is missing)
$gi = Join-Path $Proj ".gitignore"
$need = @("# --- Phase 0 additions ---", "dataset/", "*.zip", "logs/", "models/*.joblib", "data/_verify.parquet/", "venv/", ".venv/", "scripts/env.ps1")
$existing = @(); if (Test-Path $gi) { $existing = Get-Content $gi }
$toAdd = $need | Where-Object { $existing -notcontains $_ }
if ($toAdd.Count -gt 0) { Add-Content -Path $gi -Value ("`n" + ($toAdd -join "`n")); Write-Host ".gitignore: added $($toAdd.Count) rule(s)" }
# record detected JDK / Hadoop in configs\spark.yaml (project-scoped pinning)
$yaml = Join-Path $Proj "configs\spark.yaml"
if (Test-Path $yaml) {
    $c = Get-Content $yaml -Raw
    if ($jdk) { $c = $c -replace "(?m)^java_home:.*$", ("java_home: '" + $jdk + "'") }
    $c = $c -replace "(?m)^hadoop_home:.*$", ("hadoop_home: '" + $HadoopHome + "'")
    Set-Content -Path $yaml -Value $c -Encoding UTF8
    Write-Host "configs\spark.yaml updated with java_home / hadoop_home"
} else { Write-Warning "configs\spark.yaml not found (it should have been placed before running this script)." }
# session helper
$envPs1 = Join-Path $Proj "scripts\env.ps1"
@"
# Dot-source at the start of every work session:   . .\scripts\env.ps1
`$env:JAVA_HOME   = "$jdk"
`$env:HADOOP_HOME = "$HadoopHome"
`$env:Path        = "$jdk\bin;$HadoopHome\bin;" + `$env:Path
`$env:PYSPARK_PYTHON = "$Proj\venv\Scripts\python.exe"
`$env:PYSPARK_DRIVER_PYTHON = "$Proj\venv\Scripts\python.exe"
. "$Proj\venv\Scripts\Activate.ps1"
Write-Host "BIG_DATA_PROJECT environment active (JDK 17, Hadoop shim, venv)" -ForegroundColor Green
"@ | Set-Content -Path $envPs1 -Encoding UTF8
$summary["Scaffold"] = "OK"

# ---------------------------------------------------------------- 6. venv + packages
Step "6/8  Virtual environment + packages (this takes several minutes)"
$vpy = Join-Path $Proj "venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { & $py311 -m venv (Join-Path $Proj "venv") }
if (-not (Test-Path $vpy)) { Write-Error "venv creation failed"; Stop-Transcript | Out-Null; exit 1 }
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install -r (Join-Path $Proj "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Warning "pip install reported errors - see above" }
& $vpy -m pip freeze | Out-File -Encoding utf8 (Join-Path $Proj "requirements.lock.txt")
Write-Host "requirements.lock.txt written"
$summary["venv + pip"] = $(if ($LASTEXITCODE -eq 0) { "OK" } else { "CHECK" })

# ---------------------------------------------------------------- 7. verify
Step "7/8  Verifying Spark + ML stack"
$env:PYSPARK_PYTHON = $vpy; $env:PYSPARK_DRIVER_PYTHON = $vpy
$okFile = Join-Path $Proj "logs\phase0_verify.ok"
$vlog   = Join-Path $Proj "logs\phase0_verify.log"
if (Test-Path $okFile) { Remove-Item $okFile -Force }
& $vpy (Join-Path $Proj "scripts\verify_env.py") 2>&1 | Tee-Object -FilePath $vlog
if (-not (Test-Path $okFile)) {
    $txt = ""; if (Test-Path $vlog) { $txt = Get-Content $vlog -Raw }
    if ($txt -match "numpy") {
        Write-Warning "Verifier mentioned numpy - applying known fix (numpy<2) and retrying once..."
        & $vpy -m pip install "numpy<2" --quiet
        & $vpy -m pip freeze | Out-File -Encoding utf8 (Join-Path $Proj "requirements.lock.txt")
        & $vpy (Join-Path $Proj "scripts\verify_env.py") 2>&1 | Tee-Object -FilePath $vlog
    }
}
$verifyOK = Test-Path $okFile
$summary["Verifier"] = $(if ($verifyOK) { "ENVIRONMENT VERIFIED" } else { "FAILED - see logs\phase0_verify.log" })

# ---------------------------------------------------------------- 8. git
Step "8/8  Git: branch, commit, tag"
if (Get-Command git -ErrorAction SilentlyContinue) {
    git -C $Proj rev-parse --is-inside-work-tree 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        # team convention (docs/README.md): feature branches come off develop
        git -C $Proj checkout develop 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Warning "Could not switch to 'develop' (uncommitted changes?) - branching from current HEAD instead." }
        git -C $Proj checkout -B feature/phase0-environment 2>&1 | ForEach-Object { Write-Host "  $_" }
        git -C $Proj add -A
        git -C $Proj commit -m "feat: initialize project scaffold and verified Spark environment" 2>&1 | ForEach-Object { Write-Host "  $_" }
        $committed = ($LASTEXITCODE -eq 0)
        if (-not $committed) { Write-Warning "Commit failed (missing git user.name/user.email? run: git config --global user.name 'Your Name'; git config --global user.email 'you@example.com')" }
        if ($verifyOK -and $committed) { git -C $Proj tag -f v0.1-environment; Write-Host "  tagged v0.1-environment" }
        git -C $Proj log --oneline -1 | ForEach-Object { Write-Host "  HEAD: $_" }
        $summary["Git"] = $(if ($committed) { "branch feature/phase0-environment" + $(if ($verifyOK) { " + tag v0.1-environment" } else { " (no tag - verifier failed)" }) } else { "COMMIT FAILED - see warning" })
    } else { Write-Warning "Not a git repo?"; $summary["Git"] = "SKIPPED" }
} else {
    Write-Warning "git not on PATH - commit manually later (or: winget install Git.Git)."
    $summary["Git"] = "git not found"
}

# ---------------------------------------------------------------- summary
Write-Host "`n==================== PHASE 0 SUMMARY ====================" -ForegroundColor Yellow
foreach ($k in $summary.Keys) { Write-Host ("  {0,-12} {1}" -f $k, $summary[$k]) }
Write-Host "=========================================================" -ForegroundColor Yellow
if ($verifyOK) { Write-Host "PHASE 0 COMPLETE. Next session: . .\scripts\env.ps1" -ForegroundColor Green }
else { Write-Host "PHASE 0 NOT COMPLETE - send logs\phase0_verify.log for diagnosis." -ForegroundColor Red }
Stop-Transcript | Out-Null
if ($verifyOK) { exit 0 } else { exit 2 }
