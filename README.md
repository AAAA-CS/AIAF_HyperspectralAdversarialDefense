# Prerequisites
Python 3.8.16<br>
Pytorch 1.12.1<br>
Numpy 1.22.4<br>
Scipy

# Usage
1. To train the "targetmodel.py" with dataset PvaiaU ,which will generate checkpoint:'/1000_net_params_single4_Nonormalize.pkl'.  It's trained by a simple CNN classifier,you can try other targetmodel，such as ResNet, VGG.<br>
 ```asp
                        $ python targetmodel.py --dataset PaviaU --train 
   ```
  
2. Besides, the hyperspectral dataset is sourced from the link below , you can use your own dataset by matlab. I wish "SelectSample.m" could help you to select training examples. dataset link:<br>
```asp
                  https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
```

3. To train the "AIAF_train.py" to defend against adversarial examples.<br>
  ```asp
                             $ python "AIAF_train.py" --dataset PaviaU
   ```
   
4. To test with a existing model:<br>
    ```asp
                             $ python AIAF_test.py --dataset PaviaU
   ```
						  
# Related works
>[ ARN](https://github.com/dwDavidxd/ARN " ARN")<br>
[ Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch "> Torchattacks")

# Citation
If you use this code，please cite the following BibTex:<br>
@article{ <br>
  title={Attack-invariant attention feature for adversarial defense in hyperspectral image classification},<br>
  author={C.Shi,Y.Liu,M.Zhao,C.-M.Pun,Q.Miao},<br>
  journal={Pattern Recognization},<br>
  year={2023},<br>
  DOI={10.1016/j.patcog.2023.109955}<br>
}

