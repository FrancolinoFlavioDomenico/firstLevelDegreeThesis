#..\torchVenv\Scripts\activate
Start-Process  powershell {py .\src_torch\main.py --server true; Read-Host}
Start-Sleep 5


for ($i = 0; $i -lt 2; $i++) {
#  Write-Host "Loop iteration: $($i + 1)"
    Start-Process powershell { py .\src_torch\main.py --cid $i; Read-Host }
}



#Start-Process .\torchVenv\Scripts\Activate.ps1
#python