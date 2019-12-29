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

see if the fooling image looks like the origin image (low change?)
(Visualize, decided by human)

use fooler to fool model_mnist2
see if it can fool the model_mnist3 (the same model with different initial weights)

if it can fool mnist3
try to fool mnist_deep (another model)

if it cannot fool mnist3
then we come up with a method to output: class or uncertain

train mnist_defooling with fooling images run about  $28 \times 28$ epoches
see the acc change
