# On-chip-CNN-deployment-with-CMSIS-NN

## 16 bit per-tensor quantization in CMSIS NN 

https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/deploying-convolutional-neural-network-on-cortex-m-with-cmsis-nn

https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn

## Documentation

https://www.keil.com/pack/doc/CMSIS/NN/html/index.html

https://github.com/ARM-software/CMSIS_5

## Problems

1. Quantization range is not fully covered. (we don't need to represent the negative value before ReLU)
2. 32bit overflow. ('16 bit weight * 16 bit tensor + 16 bit bias' can exceed the limit of int 32)
3. how to generate the code automatically (https://github.com/majianjia/nnom)

