from lens_distortion import BrownLensDistortion
import os
from PIL import Image
from torchvision import transforms
import torch


def get_focal_length(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Extract EXIF data
        exif_data = img._getexif()

    # EXIF data is a dictionary where key is the tag number and value is the data
    if exif_data:
        # The tag for FocalLength in EXIF data is 37386
        focal_length_tag = 37386
        if focal_length_tag in exif_data:
            focal_length = exif_data[focal_length_tag]
            print(f"Focal Length: {focal_length}")
            return focal_length
        else:
            print("Focal length tag not found in EXIF data.")
    else:
        print("No EXIF data found.")

#projection = {'image_width_px': 800, 'image_height_px': 800, 'center_x_px': 400, 'center_y_px': 400}
#projection = {'image_width_px': 520, 'image_height_px': 780, 'center_x_px': 260, 'center_y_px': 390}
projection = {'image_width_px': 780, 'image_height_px': 520, 'center_x_px': 390, 'center_y_px': 260}
brown = BrownLensDistortion(k1=-0.2554, k2=0.3092, k3=-0.3180, projection=projection)
#brown = BrownLensDistortion(k1=-0.2064, k2=0.0696, k3=-0.0244, projection=projection)
#brown = BrownLensDistortion(k1=-0.1932, k2=0.0325, k3=0.0055, projection=projection)

transform = transforms.Compose([
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

def load_images_to_tensors(directory):
    tensors = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):  # Check for image file extensions
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                tensor = transform(img)
                tensors.append(tensor)

    return tensors, os.listdir(directory)

#directory_path = '/home/yd428/playaround_gaussian_platting/dataset/zip360/bonsai_dis/input/'  # Replace with your images directory path
#output_path = '/home/yd428/playaround_gaussian_platting/dataset/zip360/bonsai_dis/input_dis'  # Replace with your images directory path
directory_path = '/home/yd428/playaround_gaussian_platting/dataset/nerf_synthetic/lego/train/'  # Replace with your images directory path
output_path = '/home/yd428/playaround_gaussian_platting/dataset/nerf_synthetic/lego/train_dis'  # Replace with your images directory path
os.makedirs(output_path, exist_ok=True)
image_tensors, names = load_images_to_tensors(directory_path)

scale = [2.1920, 3.2986]
K = torch.tensor([[841.7443,   0.    , 384.    ],
                  [  0.    , 839.4974, 254.5   ],
                  [  0.    ,   0.    ,   1.    ]]).cuda()
scale = [2.778, 2.778]
K = torch.tensor([[1.1111e+03, 0.0000e+00, 4.0000e+02],
        [0.0000e+00, 1.1111e+03, 4.0000e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]]).cuda()

def save_tensor_as_image(tensor, filename):
    # Ensure the tensor is on CPU and detach it from the computation graph
    tensor = tensor.cpu().detach()

    # Convert the tensor to a PIL Image
    # Note: tensor must be cloned as it is read-only in some cases
    image = Image.fromarray((tensor.clone().numpy() * 255).astype('uint8'))

    # Save the image as a PNG file
    image.save(filename)

for image, name in zip(image_tensors, names):
    if name == '.DS_Store': continue
    #distorted_img = brown.distortImage(image[:3, :, :])
    distorted_img = brown.my_distortImage(image[:3, :, :], scale, K)
    print(name)
    try:
        save_tensor_as_image(distorted_img, os.path.join(output_path, name))
    except:
        import pdb;pdb.set_trace()



