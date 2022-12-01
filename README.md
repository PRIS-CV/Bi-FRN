# Bi-FRN

Code relaese for [Bi-Directional Feature Reconstruction Network for Fine-grained Few-shot Image Classification](https://arxiv.org/abs/2211.17161). (Accepted in AAAI-23)

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate BiFRN
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

## Train

* To train Bi-FRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/BiFRN/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/BiFRN/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/BiFRN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/BiFRN/ResNet-12
    python ./test.py
```

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch) and  [Yassine](https://github.com/yassouali/SCL), for the preliminary implementations.
