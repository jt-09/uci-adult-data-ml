$SrcFigures = Join-Path $PSScriptRoot "..\..\reports\figures"
$SrcPdf = Join-Path $PSScriptRoot "..\..\reports\overleaf\main.pdf"
$DestFigures = Join-Path $PSScriptRoot "..\figures"
$DestAssets = Join-Path $PSScriptRoot "..\assets"
New-Item -ItemType Directory -Force -Path $DestFigures, $DestAssets | Out-Null
Get-ChildItem -Path $SrcFigures -Filter "*.png" -File | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination (Join-Path $DestFigures $_.Name) -Force
}
if (Test-Path $SrcPdf) {
    Copy-Item -Path $SrcPdf -Destination (Join-Path $DestAssets "main.pdf") -Force
}
$Count = (Get-ChildItem -Path $DestFigures -Filter "*.png" -File).Count
Write-Host "Synced $Count figures into book/figures/"
if (Test-Path (Join-Path $DestAssets "main.pdf")) {
    Write-Host "Synced main.pdf into book/assets/"
}
