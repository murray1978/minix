$ErrorActionPreference = "Stop"

$ModelUrl = "https://huggingface.co/karpathy/tinyllamas/resolve/58e1696980dcbdf80b2dfe876819104a174f5e78/stories15M.bin?download=true"
$TokenizerUrl = "https://raw.githubusercontent.com/karpathy/llama2.c/refs/heads/master/tokenizer.bin"
$ExpectedModelSha256 = "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"

Write-Host "Downloading stories15M.bin..."
Invoke-WebRequest -Uri $ModelUrl -OutFile "stories15M.bin"

Write-Host "Downloading tokenizer.bin..."
Invoke-WebRequest -Uri $TokenizerUrl -OutFile "tokenizer.bin"

$Actual = (Get-FileHash -Algorithm SHA256 "stories15M.bin").Hash.ToLowerInvariant()
if ($Actual -ne $ExpectedModelSha256) {
    throw "stories15M.bin SHA256 mismatch: expected $ExpectedModelSha256, got $Actual"
}

Write-Host "Model SHA256 verified."
Get-Item "stories15M.bin", "tokenizer.bin" |
    Select-Object Name, Length, LastWriteTime
