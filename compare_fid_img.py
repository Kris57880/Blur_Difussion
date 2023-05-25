import os
import random
import matplotlib.pyplot as plt

# Set the paths to the "gt" and "pred" folders
root_dir = 'Variant_1/result'
gt_folder = root_dir+"/gt"
pred_folder = root_dir+"/pred"

# Generate a list of all file names in the "gt" folder
file_names = os.listdir(gt_folder)

# Randomly select 5 file names
selected_files = random.sample(file_names, 5)

# Create a subplot for each selected image pair
fig, axes = plt.subplots(5, 2, figsize=(10, 20))

# Loop through the selected file names
for i, file_name in enumerate(selected_files):
    # Get the file paths for the corresponding images in "gt" and "pred" folders
    gt_path = os.path.join(gt_folder, file_name)
    pred_path = os.path.join(pred_folder, file_name)
    
    # Read the images
    gt_image = plt.imread(gt_path)
    pred_image = plt.imread(pred_path)
    
    # Plot the "gt" image
    axes[i, 0].imshow(gt_image)
    axes[i, 0].set_title("GT Image")
    axes[i, 0].axis("off")
    
    # Plot the "pred" image
    axes[i, 1].imshow(pred_image)
    axes[i, 1].set_title("Pred Image")
    axes[i, 1].axis("off")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.savefig('fid_compare.png',)