$Src = Join-Path $PSScriptRoot "..\..\reports\figures"
$Dest = Join-Path $PSScriptRoot "..\figures"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Get-ChildItem -Path $Src -Filter "*.png" -File | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination (Join-Path $Dest $_.Name) -Force
}
$Count = (Get-ChildItem -Path $Dest -Filter "*.png" -File).Count
Write-Host "Synced $Count figures into book/figures/"
