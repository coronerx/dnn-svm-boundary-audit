'''
Question 1
ResNet and SVM both work in features instead of pixel space. Therefore, blended images will
suggest a feature vector that's between two clusters, indicating they won't predict randomly. 
That is, a smooth transition would happen. If a space between two digits pass through a third digit's region, 
the blending might create confusable digits.
SVM and ResNet shouldn't guess the wildly on blended digits.

Question 2
My intuitions seem to be correct by seeing both models' transition went smoothly, and they happen around
the same location. Also, no third class appeared.

Question 3
Done.

Question 4
After searching for many digit pairs, SVM blending 0 to 4, which visually
looks confusing, hallucinated to class 8 at some middle steps, and DNN stays stable.
See 0-4.png

Question 5
Two of different 1s would hallucinate to 8 in both models in middle steps, as suggested visualy.

Question 6
Taking random shots of digits from a written assignment, seems like all but 7 are
properly classified, possibly because my handwritten 7 is visually confusing without a crossbar.

'''

import argparse
import gzip
import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import sklearn
# This is for saving the trained SVMs. We could use onnx for SVMs and DNNs, but that is slightly more work.
import pickle


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=torch.nn.ReLU, stride=1):
        super(ResidualBlock, self).__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            nonlinearity(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = torch.nn.Sequential()
        # The nonlinearity after summing the residual and shortcut
        self.nonlinearity = nonlinearity()

    def forward(self, x):
        out = self.residual(x)
        x = self.shortcut(x)
        return self.nonlinearity(out + x)

class ResNet(torch.nn.Module):
    """A mostly faithful recreation of LeNet 5."""

    def __init__(self, nonlinearity = torch.nn.ReLU):
        super(ResNet, self).__init__()
        self.net = torch.nn.Sequential(
                # 5x5 convolution with 8 output feature maps
                torch.nn.Conv2d(1, 16, kernel_size=5),
                torch.nn.BatchNorm2d(16),
                nonlinearity(),
                ## Now we are working with 28x28 feature maps
                ## 3 blocks per downscale, to 14x14, 7x7, 
                ResidualBlock(16, 16),
                ResidualBlock(16, 16),
                ResidualBlock(16, 32, stride=2),
                ResidualBlock(32, 32),
                ResidualBlock(32, 32),
                ResidualBlock(32, 64, stride=2),
                ResidualBlock(64, 64),
                ResidualBlock(64, 64),
                ResidualBlock(64, 128, stride=2),
                # A single average pool to reduce all feature channels to 1x1
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 84),
                # We are not going to try to recreate the original exemplar-based function in LeNet5
                #euclidean_rbf(84, 12)
                torch.nn.Linear(84, 10),
                )
        self.decision = torch.nn.Softmax(dim=1)

        torch.nn.init.uniform_(self.net[0].weight.data, a=-1, b=1)

    def features(self, x):
        # Go through the first 14 layers to extract a feature vector of size 128
        for i in range(14):
            x = self.net[i](x)
        return x

    def forward(self, x):
        """Forward through the network."""
        y_hat = self.decision(self.net(x))
        return y_hat


def preprocess(X_train, order, device):
    # normalize and then pad to 32x32
    # Images are 0 to 1.
    # Change so the background (white) became -0.1, and the foreground (black) became 1.175
    # Multiply by 1.275 to shift expand the range, and subtract from 1.175
    preprocessed = torch.tensor(1.175 - (1.275*X_train[order])).float()
    
    # Pad 2 on every side, changing the 28x28 to 32x32
    preprocessed = torch.nn.functional.pad(preprocessed, pad=(2,2,2,2))
    # Add a channel dimension
    return preprocessed.reshape((-1, 1, 32, 32)).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first",
        required=True,
        help="first image")
    parser.add_argument(
        "--second",
        required=True,
        help="second image")
    parser.add_argument(
        "--load_dnn",
        required=True,
        default=None,
        type=str,
        help="Path to load the trained resnet model")
    parser.add_argument(
        "--load_svm",
        required=True,
        default=None,
        type=str,
        help="Path to load the pickle of the trained scikit svm.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data")
    first = np.array(Image.open(args.first))/255.0
    second = np.array(Image.open(args.second))/255.0
    # Make a blending of the images in 101 steps, from fully first to fully second
    blends = []
    for i in range(101):
        alpha = (100 - i) / 100
        blend = first * alpha + second * (1 - alpha)
        blends.append(blend)
    data = np.array(blends)


    # Don't forget that we still need to preprocess the data as was done in training
    data = preprocess(data, np.arange(data.shape[0]), device)

    print("Reloading model and SVM")
    # Create the model
    model = ResNet()
    model.load_state_dict(torch.load(args.load_dnn, map_location=torch.device("cpu"), weights_only=True))
    model.to(device)

    # Load the SVM
    with open(args.load_svm, 'rb') as infile:
        svm = pickle.load(infile)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Run the model and SVM

        print("Building test inputs for the SVM.")
        features = None
        features = model.features(data).cpu().numpy()

        print("Inference with the SVM.")
        # SVM Classification
        svm_classes = svm.predict(features)
        svm_y_hat = svm.decision_function(features)
        print(f"SVM scores {svm_y_hat}")
        print(f"SVM classes {svm_classes}")

        # DNN classification
        y_hat = model(data)
        classes = torch.argmax(y_hat, dim=1)
        print(f"DNN scores {y_hat}")
        print(f"DNN classes {classes}")
        
        class1 = int(svm_classes[0])
        class2 = int(svm_classes[-1])
        
        #plotting scores
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        steps = np.arange(101)
        
        # SVM scores
        ax1.plot(steps, svm_y_hat[:, class1], label=f'Class {class1}')
        ax1.plot(steps, svm_y_hat[:, class2], label=f'Class {class2}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('SVM Decision Score')
        ax1.set_title('SVM Scores v.s. Steps')
        ax1.legend()
        
        # DNN scores
        dnn_scores = y_hat.cpu().numpy()
        ax2.plot(steps, dnn_scores[:, class1], label=f'Class {class1}')
        ax2.plot(steps, dnn_scores[:, class2], label=f'Class {class2}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('DNN Score')
        ax2.set_title('DNN Score v.s. Steps')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # save some blend images every 25 steps
        blends_img = []
        for i in range(101):
            alpha = (100 - i) / 100
            blend = first * alpha + second * (1 - alpha)
            blends_img.append(blend)
        
        for i in (0,101,25):
            img = (255 * blends_img[i]).astype(np.uint8)
            Image.fromarray(img).save(f'blend_step_{i:03d}.png')
            print(f"step {i:03d}")
        
        print("Done.")
