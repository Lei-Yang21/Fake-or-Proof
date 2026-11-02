import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path="./modeling/fake_or_proof_model.pth"):
    """
    Load the pre-trained model from the specified path.
    """
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model
def predict_image(model, image_path):
    """
    Predict the class of the image using the loaded model.
    """
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), 
         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    )
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), float(confidence)