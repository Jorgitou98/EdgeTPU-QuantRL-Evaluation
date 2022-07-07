Create depth-filters models for CPU and TPU (quantized and compiled)
```sh
python models_depth_filters.py -ds 2 -mind 2 -maxd 8 -fs 50 -minf 8 -maxf 108 -i 1 -d 84
```

Create depth-params models for CPU and TPU (quantized and compiled)
```sh
python models_depth_params.py -dsp 2 -mindp 1 -maxdp 13 -psp 2 -minpp 15 -maxpp 23 -i 1 -d 500
```

Inference test depth-params from previously created models (CPU and TPU)
```sh
python test_depth_params.py -dsp 2 -mindp 1 -maxdp 5 -psp 2 -minpp 15 -maxpp 23 -s 10000 -b 1
```
