# Run from C:\Users\ASUS\Downloads\reuse_audit_fixed\pilot_code
# Requires same Qiskit 0.45.3 venv and existing validity-aware pilot files.
$ErrorActionPreference = 'Stop'
python .\freeze_learning_split.py verify `
  --audit-dirs ..\fixed_full ..\wave2 `
  --out-dir .\ml_split_v1
if ($LASTEXITCODE -ne 0) { throw 'Split verification failed; STOP' }

$train = (Import-Csv .\ml_split_v1\split_ledger.csv |
  Where-Object { $_.split -eq 'train' -and $_.audit_status -eq 'screened' } |
  ForEach-Object { $_.circuit_id }) -join ','
if ([string]::IsNullOrWhiteSpace($train)) { throw 'No eligible training circuits!' }

python .\build_multistep_pilot_validity_aware.py prepare `
  --audit-dirs ..\fixed_full ..\wave2 `
  --out-dir .\ml_train_prepare_v1 `
  --circuits $train `
  --max-originals 3 `
  --max-candidates 40 `
  --max-depth 5 `
  --neighbor-engine fast `
  --verify-fast-survivors
if ($LASTEXITCODE -ne 0) { throw 'Training prepare failed; STOP' }

python .\qc_training_prepare.py `
  --split-dir .\ml_split_v1 `
  --prepare-dir .\ml_train_prepare_v1
if ($LASTEXITCODE -ne 0) { throw 'Training preparation QC failed; STOP' }

Write-Host 'Training-only candidate generation and leakage QC PASS. No Routing executed.'
