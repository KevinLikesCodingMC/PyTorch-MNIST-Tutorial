"""
Test the model
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
    from torch.utils.data import DataLoader
    log("PyTorch initialization complete")
    log(f"PyTorch version: {torch.__version__}")

    # Load libs
    log("Loading libraries...")
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

    # Load MNIST test data
    log("Loading MNIST...")
    test_dataset = MNISTDataset('mnist_test.csv')
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
    )
    log("MNIST loaded")

    # Load model
    log(f"Loading model...")
    model = NeuralNetwork()
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    log(f"model loaded")

    # testing
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # show result
    accuracy = 100.0 * correct / total
    log(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
