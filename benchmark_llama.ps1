param([ValidateSet('q4km','q8')][string]$Tag='q8')
$ErrorActionPreference='Stop'
$env:HTTP_PROXY=''
$env:HTTPS_PROXY=''
$env:ALL_PROXY=''
$env:NO_PROXY='127.0.0.1,localhost'
$base='http://127.0.0.1:8093'
& curl.exe --noproxy '*' --silent --show-error --fail "$base/health"
if($LASTEXITCODE -ne 0){throw 'llama.cpp server is not ready'}
& curl.exe --noproxy '*' --silent --show-error --fail "$base/props" --output "results/llama-$Tag-props.json"
if($LASTEXITCODE -ne 0){throw 'Cannot record server configuration'}
$props=Get-Content "results/llama-$Tag-props.json" -Raw | ConvertFrom-Json
$expectedModel=if($Tag -eq 'q8'){'d1-2B-Q8_0.gguf'}else{'d1-2B-Q4_K_M.gguf'}
if(-not $props.model_path.EndsWith($expectedModel,[StringComparison]::OrdinalIgnoreCase)){
    throw "Server model does not match benchmark tag $Tag"
}
foreach($name in @('ticket','long-ticket')){
    $body="fixtures/$name-common.json"
    $prefix="results/llama-$Tag-$name"
    1..3 | ForEach-Object {
        & curl.exe --noproxy '*' --silent --show-error --fail -H 'Content-Type: application/json' --data-binary "@$body" "$base/v1/systemone" --output "$prefix-output.json"
        if($LASTEXITCODE -ne 0){throw 'Decision warmup failed'}
    }
    & curl.exe --noproxy '*' --silent --show-error --fail "$base/metrics" --output "$prefix-metrics-before.txt"
    if($LASTEXITCODE -ne 0){throw 'Metrics snapshot failed; start server with --metrics'}
    & .cache/oha.exe -n 20 -c 1 -m POST -T application/json -D $body --no-tui --output-format json -o "$prefix-oha.json" --db-url "$prefix.sqlite" "$base/v1/systemone"
    if($LASTEXITCODE -ne 0){throw 'oha benchmark failed'}
    & curl.exe --noproxy '*' --silent --show-error --fail "$base/metrics" --output "$prefix-metrics-after.txt"
    if($LASTEXITCODE -ne 0){throw 'Metrics snapshot failed'}
}
