import numpy as np
from itertools import combinations
from torchvision import datasets, transforms
from tqdm import tqdm

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./root', train=False, download=True, transform=transform)


# Get the images and labels from the MNIST dataset
images = mnist_dataset.data.numpy()
labels = mnist_dataset.targets.numpy()

# Create an empty list to store the combined images and labels
combined_images = []
combined_labels = []

# Generate all possible combinations of 4 images
image_combinations = combinations(range(len(images)), 4)

print(len(list(image_combinations))/2)

for i, in tqdm(list(image_combinations)[:len(image_combinations/2)]):
    print(i)

# Iterate over each combination
for i,combination in tqdm(enumerate(image_combinations)):
    # Get the images and labels for the current combination
    combined_image1 = np.concatenate([images[idx].reshape(28, 28) for idx in combination[:2]], axis=0)
    combined_image2 = np.concatenate([images[idx].reshape(28, 28) for idx in combination[2:]], axis=0)
    combined_image = np.concatenate([combined_image1, combined_image2], axis=1).reshape(56,56,1)

    combined_label = labels[list(combination)]

    # Convert the combined image to PIL image
    combined_image = transforms.ToPILImage()(combined_image)

    # Append the combined image and label to the dataset
    combined_images.append(combined_image)
    combined_labels.append(combined_label)

# Save the combined dataset
np.save('pvr_mnist/combined_labels.npy', combined_labels)

# Save the combined images as individual image files
for i, image in enumerate(combined_images):
    image.save(f'pvr_mnist/combined_image_{i}.png')

