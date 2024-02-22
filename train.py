"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder,precision_recall
from torchvision.transforms import v2

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.1

# Setup directories

train_dir = r"..\PCOS_dataset\train"
test_dir = r"..\PCOS_dataset\valid"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = v2.Compose([
    v2.Resize(224),  # Adjust image size as needed
    v2.CenterCrop(224),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomGrayscale(0.1),
    v2.RandomRotation(20),
    v2.RandomAffine(degrees=20,shear=20),
    v2.ToTensor(),
    v2.Normalize(mean=[0.25, 0.24, 0.24], std=[0.18, 0.18, 0.18])
])
# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.PCONet(input_shape=3,output_shape=2).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,5,0.001)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             scheduler=scheduler)

precision, recall = precision_recall.prec_recall_score(model,
                                                       test_dataloader,
                                                       loss_fn,
                                                       device)
print(f"Precision:{precision}, recall:{recall}")
