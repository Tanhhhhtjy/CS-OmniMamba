$ErrorActionPreference = 'Stop'

$workspace = 'D:/CS-OmniMamba'
$bash = 'C:/Program Files/Git/bin/bash.exe'
if (-not (Test-Path $bash)) {
	$bash = 'C:/Program Files/Git/usr/bin/bash.exe'
}
if (-not (Test-Path $bash)) {
	throw 'Git Bash not found. Please install Git for Windows.'
}

$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:ALL_PROXY = ''

$cmd = "cd $workspace; dos2unix .superpowers-installer/install-superpowers.sh >/dev/null 2>&1 || true; printf 'Y\n' | bash .superpowers-installer/install-superpowers.sh"
& $bash -lc $cmd
