import os

directory = 'sphere/beta=1-half'  # replace with your directory path

for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        parts = filename.split('-')
        # pad with zeros up to 5 digits
        parts[-1] = parts[-1].zfill(10)  # '.zfill(10)' will add leading zeros to make the number 10 digits long
        new_filename = '-'.join(parts)
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))