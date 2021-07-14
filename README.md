# On-chip-CNN-deployment-with-CMSIS-NN

1. check out official tutorial and documentations
2. check out `Background_Knowledge.ipynb`
3. check out `Quantization.ipynb`
4. run `CMSIS_NN_PC_simulator` in Visual Studio to verify the results
5. deploy the model on Cortex M4 boards

## Per-tensor quantization in CMSIS NN (official tutorial)

https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/deploying-convolutional-neural-network-on-cortex-m-with-cmsis-nn

https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn

## Documentation

https://www.keil.com/pack/doc/CMSIS/NN/html/index.html

https://github.com/ARM-software/CMSIS_5

## Problems

1. Quantization range is not fully covered. (we don't need to represent the negative value before ReLU)
2. 32bit overflow. ('16 bit weight * 16 bit tensor + 16 bit bias' can exceed the limit of int 32)
3. how to generate the code automatically (https://github.com/majianjia/nnom)

## STFT 

`ARM_STFT_ISTFT.c`: STFT and ISTFT for ARM boards (same with libosa's implementation)

