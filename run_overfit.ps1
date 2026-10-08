param(
    [ValidateSet('single', 'five')][string]$Experiment = 'single',
    [int]$Steps = 1000,
    [double]$LearningRate = 0.003,
    [string]$Output = '',
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskWslPath = (& wsl -d cold-surge-ubuntu-24.04 -- wslpath -a $taskRoot).Trim()
if (!$Output) { $Output = "runs/hashgrid_aligned_${Experiment}_${Steps}" }
$trainArgs = @('-m', 'geobc.train', '--data', 'data/prepared_multifield', '--backend', 'triton', '--device', 'cuda',
    '--steps', "$Steps", '--learning-rate', "$LearningRate", '--nodes-per-case', '16384',
    '--eval-every', '100', '--target-error', '0', '--output', $Output)
if ($Experiment -eq 'single') { $trainArgs += @('--case-keys', '10_30/ver', '--input-mode', 'xyz') }
else { $trainArgs += @('--case-keys', '10_30/ver', '10_421/hor', '10_460/dia', '10_492/tor', '10_58/ver', '--input-mode', 'both') }
if ($Resume) { $trainArgs += @('--resume', $Resume) }
& wsl -d cold-surge-ubuntu-24.04 --cd $taskWslPath -- /home/research/.venvs/geobc/bin/python @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
