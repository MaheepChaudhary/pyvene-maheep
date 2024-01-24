from itertools import combinations
from torchvision import datasets, transforms
import os
from tqdm.auto import tqdm

# Set the transform for the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./root', train=False, download=True, transform=transform)

# Get the number of images in the dataset
num_images = len(mnist_dataset)

# Set the batch size for how many combinations to write at a time
batch_size = 10000  # Adjust this number based on your system's capabilities

# Open a file to write the combinations
with open('image_combinations.txt', 'w') as file:
    # Create a generator for all possible combinations of 4 images
    image_combinations = combinations(range(num_images), 4)

    # Initialize a counter for the current batch
    current_batch = []

    for i, combo in tqdm(enumerate(image_combinations)):
        # Add the current combination to the batch
        current_batch.append(combo)

        # If we've reached the batch size, write the current batch to the file
        if (i + 1) % batch_size == 0:
            # Write the current batch of combinations to the file
            file.writelines(f"{','.join(map(str, combination))}\n" for combination in current_batch)
            # Clear the current batch
            current_batch = []

    # Write any remaining combinations to the file
    if current_batch:
        file.writelines(f"{','.join(map(str, combination))}\n" for combination in current_batch)
