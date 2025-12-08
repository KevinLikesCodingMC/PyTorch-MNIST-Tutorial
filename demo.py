"""
Visual Demo of Handwritten Digit Recognition using CNN
Press Left Button to draw
Press key R to reset canvas
"""

import datetime

# print logs
def log(message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] %s' % (timestamp, message))

# Load Pytorch
log("Initializing PyTorch...")
import torch
import torch.nn.functional as F
log("PyTorch initialization complete")
log(f"PyTorch version: {torch.__version__}")

# Load libs
log("Loading libraries...")
import pygame.freetype
import numpy as np
import sys
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

# move model
log(f"Loading model...")
model = NeuralNetwork()
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
log(f"model loaded")

# init pygame
log(f"Initializing...")
pygame.init()
screen = pygame.display.set_mode((1500, 800))
pygame.display.set_caption('Recognize Numbers')

# pygame colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GREY = (200, 200, 200)
DARK_GREY = (50, 50, 50)

# 600 x 600 canvas
canvas = np.zeros((600, 600))
image = np.zeros((28, 28))
scl = 600 / 28

# Shrink
def shrink(canvas):
    canvas_tensor = torch.from_numpy(canvas).float()
    canvas_tensor = canvas_tensor.unsqueeze(0).unsqueeze(0)
    # blur
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]
    kernel = torch.tensor(kernel, dtype=torch.float32) / 16.0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(canvas_tensor, kernel, padding=1)
    # 600 x 600 -> 28 x 28
    scaled = F.interpolate(blurred, size=(28, 28), mode='bilinear', align_corners=False)
    result = scaled.squeeze().numpy()
    return result

# recognition
def recognize(image):
    image_torch = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image_torch = image_torch.to(device)
    with torch.no_grad():
        output = model(image_torch)
        output = torch.softmax(output, dim=1)
        output = output.cpu().numpy()
    return output


# drawing state
drawing = False

last_call_time = 0
interval = 50

# use the default font
font = pygame.freetype.Font(None, 40)

# CNN output
output = recognize(image)

log("Initialization complete")

while True:
    # events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # switch drawing state
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        # reset canvas
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                canvas = np.zeros((600, 600))
                image = np.zeros((28, 28))

    # clear screen
    screen.fill(WHITE)

    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_x -= 100
        mouse_y -= 100
        r = 22
        # drawing a circle
        for y in range(max(0, mouse_y - r), min(mouse_y + r, 600)):
            for x in range(max(0, mouse_x - r), min(mouse_x + r, 600)):
                if (mouse_x - x) ** 2 + (mouse_y - y) ** 2 <= r ** 2:
                    canvas[y, x] = 1
        # 600 x 600 -> 28 x 28
        image = shrink(canvas)
        # 28 x 28 -> 10
        output = recognize(image)

    # draw the 28 x 28 image
    for y in range(28):
        for x in range(28):
            val = int(image[y][x] * 255)
            val = max(0, min(255, val))
            clr = (val, val, val)
            rect_x = int(x * scl + 100)
            rect_y = int(y * scl + 100)
            rect_size = int(scl) + 1
            pygame.draw.rect(screen, clr, (rect_x, rect_y, rect_size, rect_size))

    # draw inner border
    pygame.draw.rect(screen, DARK_GREY, (250, 200, 300, 400), 3)
    # drawing outer border
    pygame.draw.rect(screen, LIGHT_GREY, (100, 100, 600, 600), 5)

    # draw number bars
    for number in range(10):
        font.render_to(screen, (800 + number * 60, 650), str(number), BLACK)
        rect_x = 795 + number * 60
        rect_height = 500 * float(output[0][number])
        rect_y = 620 - rect_height
        pygame.draw.rect(screen, BLACK, (rect_x, rect_y, 30, rect_height))

    # update screen
    pygame.display.update()
