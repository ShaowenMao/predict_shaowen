[CmdletBinding()]
param(
    [string]$Document = "",
    [string]$OutputDirectory = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$pnasDir = Split-Path -Parent $PSScriptRoot
$paperDir = Split-Path -Parent $pnasDir
if ([string]::IsNullOrWhiteSpace($Document)) {
    $Document = Join-Path $pnasDir "manuscript.tex"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $pnasDir "build"
}

$Document = [System.IO.Path]::GetFullPath($Document)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$latexmkCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\latexmk.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX 2.9\miktex\bin\x64\latexmk.exe")
)

$pathLatexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($null -ne $pathLatexmk) {
    $latexmkCandidates += $pathLatexmk.Source
}

$latexmk = $latexmkCandidates |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1

if ([string]::IsNullOrWhiteSpace($latexmk)) {
    throw "latexmk was not found. Install MiKTeX or TeX Live, then retry."
}

$pdflatex = Join-Path (Split-Path -Parent $latexmk) "pdflatex.exe"
if (-not (Test-Path -LiteralPath $pdflatex)) {
    $pathPdfLaTeX = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($null -eq $pathPdfLaTeX) {
        throw "pdflatex was not found alongside latexmk or on PATH."
    }
    $pdflatex = $pathPdfLaTeX.Source
}

$documentName = Split-Path -Leaf $Document
$enginePath = $pdflatex.Replace("\", "/")
$engineCommand = '"{0}" --disable-installer %O %S' -f $enginePath
$previousBibInputs = $env:BIBINPUTS
$pathSeparator = [System.IO.Path]::PathSeparator
$env:BIBINPUTS = if ([string]::IsNullOrWhiteSpace($previousBibInputs)) {
    "$paperDir$pathSeparator"
}
else {
    "$paperDir$pathSeparator$previousBibInputs"
}

Push-Location $pnasDir
try {
    if ($Clean) {
        & $latexmk -C "-pdflatex=$engineCommand" "-outdir=$OutputDirectory" $documentName
    }
    else {
        & $latexmk -pdf "-pdflatex=$engineCommand" `
            -interaction=nonstopmode -file-line-error -synctex=1 `
            "-outdir=$OutputDirectory" $documentName
    }

    if ($LASTEXITCODE -ne 0) {
        throw "PNAS LaTeX build failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:BIBINPUTS = $previousBibInputs
    Pop-Location
}
