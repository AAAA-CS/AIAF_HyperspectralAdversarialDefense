# Prerequisites
Python 3.8.16<br>
Pytorch 1.12.1<br>
Numpy 1.22.4<br>
Scipy

# Usage
1. To train the "targetmodel.py" with dataset PvaiaU ,which will generate targetmodel checkpoint:'/1000_net_params_single4_Nonormalize.pkl'.
                      $ python targetmodel.py --dataset PaviaU --train
  it's trained by a simple CNN classifier,you can try other targetmodel，such as ResNet，VGG.
  
2. Besides, the hyperspectral dataset is sourced from the link below , you can use your own dataset by matlab. I wish "SelectSample.m" could help you to select training examples. dataset link:
```asp
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
```

3. To train the "AIAF_train.py" to defend against adversarial examples.
                          $python "AIAF_train.py" --dataset PaviaU
4. To test with a existing model:
                          $python AIAF_test.py --dataset PaviaU
						  
# Related works
>[ ARN](https://github.com/dwDavidxd/ARN " ARN")<br>
[ Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch "> Torchattacks")

# Citation
If you use this code，please cite the following BibTex:
@article{ 
  title={Attack-invariant attention feature for adversarial defense in hyperspectral image classification},
  author={C.Shi,Y.Liu,M.Zhao,C.-M.Pun},
  journal={Pattern Recognization},
  year={2023}
}
