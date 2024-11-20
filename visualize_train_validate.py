import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

checkpoint_path = 'F:\\Segmentation\\results\\checkpoints\\model_epoch_100.pth' # Set x.pth to number of epochs
test_image_path = 'F:\\DLC_thermography-JLanden-2024-07-09\\labeled-data\\06-24-24_3400_paired_29\\img054787.png'


# Convert image to tensor
def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

def draw_boxes(image, prediction):
    image = transforms.ToPILImage()(image.cpu())
    draw = ImageDraw.Draw(image)
    for element in range(len(prediction[0]['boxes'])):
        boxes = prediction[0]['boxes'][element].cpu().numpy()
        score = prediction[0]['scores'][element].cpu().numpy()
        if score > 0.5:  # Set a threshold
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red")
    return image

# Assuming you're using a Mask R-CNN with a ResNet50 FPN backbone
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
num_classes = 3  # Adjust this to your number of classes

# box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# mask predictor
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, num_classes)

# Load the model weights & move to the appropriate device (Either GPU or CPU)
model.load_state_dict(torch.load(checkpoint_path))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Example image path
image = transform_image(test_image_path).to(device)

# Put model in evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    prediction = model([image])

# The prediction will contain 'boxes', 'labels', and 'scores' for the detections
print(prediction)

# Draw boxes on the image
output_image = draw_boxes(image.squeeze(0), prediction)

# Display the image
plt.imshow(output_image)
plt.show()