from PIL import Image
import os
import random


original_images_folder = r"G:\dataset\detection\data_vocnorm\train\JPEGImages"
snow_masks_folder = r"H:\dataset\Snow100K-testset\media\jdway\GameSSD\overlapping\test\Snow100K-M\mask"
output_folder = r"G:\dataset\detection\data_vocsnow\train\JPEGImages"

for filename in os.listdir(original_images_folder):
    if filename.endswith(".jpg"):
        original_image_path = os.path.join(original_images_folder, filename)
        original_image = Image.open(original_image_path)
        width, height = original_image.size

        snow_mask_filename = random.choice(os.listdir(snow_masks_folder))
        snow_mask_path = os.path.join(snow_masks_folder, snow_mask_filename)
        snow_mask = Image.open(snow_mask_path)

        snow_mask_resized = snow_mask.resize((width, height)).convert("L")

        result_image = Image.new("RGB", (width, height), (0, 0, 0))
        result_image.paste(original_image, (0, 0))
        result_image.paste(snow_mask_resized, (0, 0), snow_mask_resized)

        output_path = os.path.join(output_folder, filename)
        result_image.save(output_path)
