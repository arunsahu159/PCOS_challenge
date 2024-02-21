# PCOS_challenge
Classification of Poly cystic ovary syndrome images.
 ## Dataset
  The dataset used for training the classification model can be obtained from the following source: [PCOS classification Dataset (PCOS_dataset)](https://zenodo.org/records/10430727). The dataset was first pre-processed and images were segregated into their respective classes. Then these images were augmented to increase the dataset size. The augmented dataset was then split into train and validation in 80:20 ratio.


# Team name: Arun Ranjan Sahu
Contact Number: 9679166756 <br>
Email address: arunsahu159@gmail.com

  ## Model
- The architecture consists of Five convolution blocks, each consisting of a convolution layer followed by a max-pooling layer.
- The kernel size of each convolutional layer is 3x3, the activation function is SiLU, and the stride was 1.
- The initial convolutional layer’s input shape was (224,224,3). All the maximum pooling layers had a 2x2 pool size and a stirde of 2 .
- To extract features, the number of filters for first through fifth convolutional layers were 32, 32, 64, 64, and 128 respectively.
- The model's initial weights were set using [He initialization](https://pytorch.org/docs/stable/nn.init.html).
- Fig. 1 depicts a simplified schematic of PCONet (Polycystic Ovary Syndrome Network).
- As an example, the first convolutional block is depicted in Fig. 2. Such five convolutional blocks were utilized with various parameters.
![modeL](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/6d4219b3-9cdc-4d62-b73e-4875a710389c) <p align="center"> **Figure 1.** </p>

![Model_diagram](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/3ad638ca-1d69-4490-9263-bbc1a2fd4c70) <p align="center"> **Figure 2.** </p>

 ### Achieved results on validation dataset:
 -  | Metric    | value  |
    |-----------|------  |
    | Accuracy  |84.28   |
    | Precision |100.00  |
    | Recall    |91.99   |
