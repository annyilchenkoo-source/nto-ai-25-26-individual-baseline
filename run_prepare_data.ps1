# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

# Run prepare_data using poetry
poetry run python -m src.baseline.prepare_data

