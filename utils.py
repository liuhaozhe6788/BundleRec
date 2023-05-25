import os

def find_ckpt_file(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Iterate through the files and find the first ckpt file
    for file in files:
        if file.endswith('.ckpt'):
            return os.path.join(directory, file)

    # If no ckpt file is found, return None or raise an exception
    return None  # or raise an exception