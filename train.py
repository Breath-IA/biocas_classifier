from pipeline_api import load_dataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision.models import efficientnet_b0
from tqdm import tqdm

def train(test_dataset, train_dataset):
    N_EPOCHS = 100

    model = efficientnet_b0(progress=True)

    new_in_channels = 1

    # Replace the first convolutional layer
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        new_in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)

    model.to("cuda")

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for features, labels in train_loader:

            features, labels = features.to("cuda"), labels.to("cuda")
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_acc = 100 * correct / total
            print(current_acc)


            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.2f}%"
            })

    print("Dataset listo para training.")
    print(f"  Train : {len(train_loader)} muestras")
    print(f"  Val   : {len(val_loader)} muestras")
    print(f"  Shape : {train_dataset.feature_shape}  (C, n_mels, T)")
    print(f"  Clases: {train_dataset.num_classes}")

if __name__=="__main__":
    test_dataset = load_dataset("data/processed_v3/test")
    train_dataset = load_dataset("data/processed_v3/train")

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(test_dataset,   batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Inspeccionar un batch
    features, labels = next(iter(train_loader))
    print(f"features : {features.shape}   dtype={features.dtype}")
    print(f"labels   : {labels.shape}     dtype={labels.dtype}")
    print(f"Pesos para CrossEntropyLoss: {train_dataset.class_weights}")

    train(test_dataset, train_dataset)