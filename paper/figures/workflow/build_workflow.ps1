param(
    [switch]$Clean,
    [switch]$RefreshScientificAssets
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ProjectRoot 'build'
$SourceFile = Join-Path $ProjectRoot 'workflow_overview.tex'
$FinalPdf = Join-Path $ProjectRoot 'workflow_overview.pdf'
$PreviewBase = Join-Path $ProjectRoot 'workflow_overview_preview'

if ($Clean) {
    if (Test-Path -LiteralPath $BuildDir) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
    Remove-Item -LiteralPath $FinalPdf -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath ($PreviewBase + '.png') -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Push-Location $ProjectRoot
try {
    # Regenerate the Panel 1 Pc/Kr asset with its original CairoMakie code.
    $Panel1Renderer = Join-Path $ProjectRoot 'render_panel1_pc_kr_curves.jl'
    $Panel1Asset = Join-Path $ProjectRoot 'assets\pc_kr_curves.pdf'
    $Panel1Data = if ($env:PREDICT_WORKFLOW_PANEL1_SOURCE) {
        $env:PREDICT_WORKFLOW_PANEL1_SOURCE
    }
    else {
        'D:\codex_gom\UQ_workflow\representative_stratigraphy_schematic\predict_permeability_distributions\s02_c003\toc_components\source_data\reservoir_ready_s02_c003_case01.mat'
    }
    $RenderPanel1 = $RefreshScientificAssets -or -not (Test-Path -LiteralPath $Panel1Asset)
    if (-not $RenderPanel1 -and (Test-Path -LiteralPath $Panel1Data)) {
        $NewestInput = (Get-Item -LiteralPath $Panel1Renderer).LastWriteTimeUtc
        $DataTime = (Get-Item -LiteralPath $Panel1Data).LastWriteTimeUtc
        if ($DataTime -gt $NewestInput) {
            $NewestInput = $DataTime
        }
        $RenderPanel1 = (Get-Item -LiteralPath $Panel1Asset).LastWriteTimeUtc -lt $NewestInput
    }
    if ($RenderPanel1) {
        if (-not (Test-Path -LiteralPath $Panel1Data)) {
            throw "Panel 1 Pc/Kr source data not found: $Panel1Data"
        }
        & julia --startup-file=no $Panel1Renderer
        if ($LASTEXITCODE -ne 0) {
            throw "Panel 1 Pc/Kr rendering failed with exit code $LASTEXITCODE."
        }
    }

    # Rebuild figure derivatives so embedded scientific graphics have clean,
    # transparent backgrounds and manuscript-scale crops.
    & python (Join-Path $ProjectRoot 'prepare_left_panel_assets.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Left-panel asset preparation failed with exit code $LASTEXITCODE."
    }

    & python (Join-Path $ProjectRoot 'prepare_middle_panel_assets.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Middle-panel asset preparation failed with exit code $LASTEXITCODE."
    }

    $Panel3MigrationAsset = Join-Path $ProjectRoot `
        'assets\derived\panel3_migration_cross_sections_transparent.pdf'
    if ($RefreshScientificAssets -or -not (Test-Path -LiteralPath $Panel3MigrationAsset)) {
        & python (Join-Path $ProjectRoot 'render_panel3_migration_cross_sections.py') `
            --output-stem panel3_migration_cross_sections_transparent
        if ($LASTEXITCODE -ne 0) {
            throw "Panel 3 migration asset preparation failed with exit code $LASTEXITCODE."
        }
    }

    $Panel3MigrationUqAsset = Join-Path $ProjectRoot `
        'assets\derived\panel3_uq_migration.pdf'
    $Panel3ContainmentUqAsset = Join-Path $ProjectRoot `
        'assets\derived\panel3_uq_containment.pdf'
    if ($RefreshScientificAssets -or
        -not (Test-Path -LiteralPath $Panel3MigrationUqAsset) -or
        -not (Test-Path -LiteralPath $Panel3ContainmentUqAsset)) {
        & python (Join-Path $ProjectRoot 'prepare_panel3_uq_assets.py')
        if ($LASTEXITCODE -ne 0) {
            throw "Panel 3 UQ asset preparation failed with exit code $LASTEXITCODE."
        }
    }

    # A single pdflatex pass is sufficient because this standalone figure has
    # no cross-references. It is also more reliable than latexmk on MiKTeX when
    # packages are being initialized for the first time.
    & pdflatex --enable-installer -interaction=nonstopmode -halt-on-error `
        -file-line-error "-output-directory=$BuildDir" $SourceFile
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX build failed with exit code $LASTEXITCODE."
    }

    Copy-Item -LiteralPath (Join-Path $BuildDir 'workflow_overview.pdf') `
        -Destination $FinalPdf -Force

    $PdfToPpm = (Get-Command pdftoppm -ErrorAction Stop).Source
    if ([System.IO.Path]::GetExtension($PdfToPpm) -eq '.cmd') {
        $Candidate = [System.IO.Path]::GetFullPath((Join-Path `
            (Split-Path -Parent $PdfToPpm) `
            '..\..\native\poppler\Library\bin\pdftoppm.exe'))
        if (Test-Path -LiteralPath $Candidate) {
            $PdfToPpm = $Candidate
        }
    }

    # Render a fixed-width review image while preserving the caption-aware
    # aspect ratio of the vector PDF.
    & $PdfToPpm -png -singlefile -scale-to-x 963 -scale-to-y -1 `
        $FinalPdf $PreviewBase
    if ($LASTEXITCODE -ne 0) {
        throw "PDF preview rendering failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Built: $FinalPdf"
Write-Host "Preview: $($PreviewBase + '.png')"
