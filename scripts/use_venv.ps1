$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Missing venv at $venvActivate"
}

$env:HF_HOME = Join-Path $repoRoot ".hf-home"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:PIP_CACHE_DIR = Join-Path $repoRoot ".pip-cache"

New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_DATASETS_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:HUGGINGFACE_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:TRANSFORMERS_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

. $venvActivate

Write-Host "Activated KG-LLM-TUNE venv at $repoRoot with Hugging Face and pip caches under the repo."
