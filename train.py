"""
Train the models
"""

def main():
    # print logs
    import datetime
    def log(message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('[%s] %s' % (timestamp, message))

    # Load Pytorch
    log("Initializing PyTorch...")
    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data import DataLoader
    log("PyTorch initialization complete")
    log(f"PyTorch version: {torch.__version__}")

    # Load libs
    log("Loading libraries...")
    import matplotlib.pyplot as plt
    from utils import MNISTDataset
    from model import NeuralNetwork
    log("Libraries loaded")

    # Check devices
    log("Checking available compute devices...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"GPU detected: {torch.cuda.get_device_name(0)}")
        log(f"Available GPU count: {torch.cuda.device_count()}")
        log(f"Selected device: GPU (cuda:0) - {torch.cuda.get_device_name(0)}")
        log(f"CUDA version: {torch.version.cuda}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        log("No GPU available, using CPU for computation")
        log(f"Selected device: CPU")

    # Load MNIST train data
    log("Loading MNIST...")
    train_dataset = MNISTDataset('mnist_train.csv')
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )
    log("MNIST loaded")

    # move model
    log(f"Moving model to {device}...")
    model = NeuralNetwork().to(device)
    log(f"Model successfully moved to {device}")

    # setup model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    log("Model setup complete")

    losses = []
    accuracies = []

    # training
    log("Starting training...")

    # 10 epoches
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # load data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # calc loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # collect
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        # collect
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        log('Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))

    log("Training complete")

    # save model
    log("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'accuracies': accuracies
    }, 'model.pth')
    log("Model saved")

    # show result
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-', linewidth=2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
