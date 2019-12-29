Li Jiasen's readme:

```bash
python inp.py
```

run FMNIST or MNIST

MNIST 99.5

FMNIST 90

This is ResNet-6




Our task

train mnist2/3/deeper (Li Jiasen)
acc > 99.2

train a fooler for specific model(mnist2/3)  (Li Jiasen)
can fool 10 % of all data

see if the fooling image looks like the origin image (low change)

use fooler to fool model_mnist2
see if it can fool the model_mnist3 (the same model with different initial weights)

if it can fool mnist3
try to fool mnist_deep (another model)(ResNet-7)

if it cannot fool mnist3
then we come up with a method to output: class or uncertain

train mnist_defooling with fooling images run about  $28 \times 28$ epoches
see the acc change


白盒fooling 
1.具体的参数的值是否影响fooling(同样的网络，不同的初始值)(此实验需要重复多次)
2.如果有效，那么对于不同结构和相同数据是否有效
3.如果无效，那么是否对于不同结构，同时被fooling的概率相当小？
4.用自己的fooling image去训练
