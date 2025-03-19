import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image

def list_files_with_suffix(folder, suffix):
    # List to store matching files
    matching_files = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        # Check if it's a file and ends with the specified suffix
        if os.path.isfile(file_path) and filename.endswith(suffix):
            matching_files.append(filename)

    return matching_files


def process_fits_and_save_png(fits_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all FITS files in the input folder
    fits_files = [f for f in os.listdir(fits_folder) if f.endswith('.fits')]
    
    # Process each FITS file
    total = len(fits_files)
    for index,fits_file in enumerate(fits_files):
        if index %10==0:print("{}/{}".format(index,total))
        fits_path = os.path.join(fits_folder, fits_file)
        
        # Read the FITS file
        with fits.open(fits_path) as hdul:
            # Assume the image data is in the primary HDU (index 0)
            image_data = hdul[0].data
            
            # Check if the data is 2D (2D images are assumed to be what we're working with)
            if image_data.ndim != 2:
                print(f"Skipping non-2D FITS file: {fits_file}")
                continue
            
            # Apply zscale to the image data for contrast enhancement
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(image_data)
            
            # Apply Z-scale normalization (clipping values between vmin and vmax)
            image_data = np.clip(image_data, vmin, vmax)
            image_data = (image_data - vmin) / (vmax - vmin) * 255  # Scale to 0-255 range
            
            # Convert the image data to an unsigned 8-bit integer (for saving as PNG)
            image_data = image_data.astype(np.uint8)
            
            # Create a PIL Image object from the numpy array
            pil_image = Image.fromarray(image_data)
            
            # Output file path (same name as FITS file, but with .png extension)
            output_png_path = os.path.join(output_folder, f"{os.path.splitext(fits_file)[0]}.png")
            
            # Save the image as PNG using Pillow
            pil_image.save(output_png_path, format='PNG')
            
            # print(f"Processed {fits_file} and saved as {output_png_path}")

if __name__ == "__main__":
    # Example usage:
    fits_folder = '/home/kevin/RTObjDet/testfits'  # Folder containing FITS files
    output_folder = '/home/kevin/RTObjDet/datasetv2/test'   # Folder where PNG images will be saved

    process_fits_and_save_png(fits_folder, output_folder)