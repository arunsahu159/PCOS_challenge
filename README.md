# PCOS_challenge
Classification of Poly cystic ovary syndrome images.
 ## Dataset
  The dataset used for training the classification model can be obtained from the following source: [PCOS classification Dataset (PCOS_dataset)](https://zenodo.org/records/10430727). The dataset was first pre-processed and images were segregated into their respective classes. Then these images were augmented to increase the dataset size. The augmented dataset was then split into train and validation in 80:20 ratio. The augmented dataset can be found [here](https://drive.google.com/file/d/1uMXvIqSobsJfuSVmwW_KCWqcaGFcZXol/view?usp=sharing)
 ## How to run

Clone this repository to your local machine
```bash
git clone <>
cd PCOS_challenge
python train.py
```

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

![modeL-2](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/4e6b82d3-36ae-4e86-8da1-30a3a09c21a5) <p align="center"> **Figure 2.** </p>

 ### Achieved results on validation dataset:

 
 -  | Metric    | value  |
    |-----------|------  |
    | Accuracy  |84.28   |
    | Precision |100.00  |
    | Recall    |91.99   |

- #### Pictures of 5 best frames selected from validation dataset

![image_12_pred_0_gt_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/84224589-01c9-47c1-99c1-1c86260e8916)
![image_31_pred_0_gt_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/73ee7248-ce6a-47d7-af52-a329705933aa)
![image_254_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/5a113ea3-c806-4e30-a55f-e622fea1eabb)
![image_299_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/0da56c02-03af-47a0-8284-d8672c5ed5fc)
![image_306_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/956b10d4-365a-4ba3-a3c3-a935979b68c4)


- ### Picture of achieved interpretability plots from validation dataset

 
![image_12_pred_0_gt_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/7b01fec3-ad6a-4f7b-982a-7d1b54cec8bd)
![image_31_pred_0_gt_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/bff1672b-93f7-4d6b-944c-55b16a2d15e2)
![image_254_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/e6e18ce6-8673-4728-8435-bf94ddafc082)
![image_299_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/2a35adb7-d23c-4bcb-8b78-96e5982700bf)
![image_306_pred_1_gt_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/6810a62a-76f0-4491-b1a7-34ecdf0d0718)

### Achieved results on test dataset:

![1_pred_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/3673f1ba-8dd2-43f6-a4ec-6090f30e9a07)
![178_pred_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/5234aaaa-95da-4afc-a682-ba44eaffaa8e)
![211_pred_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/493016da-46f1-4c39-a9f5-0551ec0faa01)
![235_pred_0](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/65d0bb50-b0dc-4571-8e90-f5d18d43c13c)
![289_pred_1](https://github.com/arunsahu159/PCOS_challenge/assets/61779161/8edb4895-8179-4576-92bb-694fae79f910)


