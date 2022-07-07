Create depth-param models (TFLITE) for CPU and TPU (quantized and compiled)
```sh
python models_depth_params.py -dsp 2 -mindp 1 -maxdp 13 -psp 2 -minpp 15 -maxpp 23 -i 1 -d 500
```

Inference test from previously created models (CPU and TPU)s
```sh
python test_depth_params.py -dsp 2 -mindp 1 -maxdp 5 -psp 2 -minpp 15 -maxpp 23 -s 10000 -b 1
```
