export PATH_TO_DATA=~/data

if [[ $(hostname) == "oon17" ]]; then
    export PATH_TO_DATA=/home/prem/shared/data/
fi

if [[ $(hostname) == "oon19" ]]; then
    export PATH_TO_DATA=/home/prem/shared/data/
fi

if [[ $(hostname) == "lucas-ssound-trt-vm" ]]; then
    export PATH_TO_DATA=~/data
fi

if [[ $(hostname) == "a100-ssound" ]]; then
    export PATH_TO_DATA=~/data
fi

if [[ $(hostname) == "oon25" ]]; then
    export PATH_TO_DATA=/data
fi

if [[ $(hostname) == "macbook-pro-2.lan" ]]; then
    export PATH_TO_DATA=~/data
fi

if [[ $(hostname) == "oon11" ]]; then
    export PATH_TO_DATA=/data2/syncthing_lucas/data
fi

if [[ $(hostname) == "oon12" ]]; then
    export PATH_TO_DATA=/data
fi
if [[ $(hostname) == "oon26" ]]; then
    export PATH_TO_DATA=/data
fi
