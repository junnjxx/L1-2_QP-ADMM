import os
from PIL import Image

def convert_images_to_grayscale(data_folder_path, new_data_folder_path):
    """
    Converts all images in the given folder to grayscale and saves them with the same name.

    Args:
        data_folder_path (str): Path to the folder containing the images.

    Returns:
        None
    """
    if not os.path.exists(data_folder_path):
        print(f"The folder '{data_folder_path}' does not exist.")
        return

    # List all files in the folder
    for file_name in os.listdir(data_folder_path):
        file_path = os.path.join(data_folder_path, file_name)
        new_file_path =  os.path.join(new_data_folder_path, file_name)

        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Convert to grayscale
                    gray_img = img.convert('L')

                    # Save the grayscale image, overwriting the original file
                    gray_img.save(new_file_path)
                    print(f"Converted {file_name} to grayscale.")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
data_folder_path = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_cifar10/'
new_data_folder_path = '/home/ubuntu/xlj/opt/bigbatch/cifar10_datasets/train_cifar10_gray/'
convert_images_to_grayscale(data_folder_path, new_data_folder_path)