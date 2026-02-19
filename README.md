# Image Classification with CNN (CIFAR-10)



A simple Image Classification project using PyTorch.  

The model is trained on CIFAR-10 dataset and supports training, checkpoint saving, inference, and TensorBoard logging.



---



## Project Structure



```

Image-Classification/

│

├── checkpoints/        # Saved models 

├── images/             # Demo images for inference

├── src/

│   ├── cifar10.py      # Dataset loader

│   ├── models.py       # CNN model definition

│   ├── train.py        # Training script

│   └── inference.py    # Inference script

│

├── requirements.txt

└── README.md

```



---



## Model



- Custom CNN

- Trained on CIFAR-10

- Input size: 224x224

- Optimizer: Adam

- Loss: CrossEntropyLoss



---



## Installation



### Clone repo



```bash

git clone https://github.com/thienlongdev/image-classification.git

cd image-classification

```



### Create virtual environment



```bash

python -m venv .venv

.venv\\Scripts\\activate   # Windows

```



### Install dependencies



```bash

pip install -r requirements.txt

```



---



## Training



Run:



```bash

python src/train.py

```



Or with custom arguments:



```bash

python src/train.py --epochs 50 --batch-size 16

```



Checkpoints will be saved in:



```

checkpoints/

```



---



## TensorBoard



Run:



```bash

tensorboard --logdir tensorboard

```



Open in browser:



```

http://localhost:6006

```



---



## Inference



Run prediction on demo images:



```bash

python src/inference.py

```



Images are stored in:



```

images/

```



---



## Demo Images



Sample input images:



- airplane

- bird

- frog

- horse

- truck



Example prediction output:



```

Image: airplane.jpg → Predicted class: airplane

Image: frog1.jpg → Predicted class: frog

```



---



## Requirements

Install with:



```bash

pip install -r requirements.txt

```



---



## Features



- Custom CNN  

- Checkpoint saving (best \& last)  

- TensorBoard logging  

- Inference script  

- Clean project structure  



---



## Future Improvements



- Add ResNet18 option

- Add validation accuracy plot

- Add confusion matrix

- Deploy with Streamlit



---


