# imports
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from src.resnet import ResNet18

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =fastAPI app
app = FastAPI(title="ResNet-18 from Scratch — 79% Accuracy")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <div style="font-family: Arial; text-align: center; margin: 40px;">
        <h1 style="color: #1DA1F2;">ResNet-18 from Scratch</h1>
        <p><strong>79% Accuracy on CIFAR-10</strong> — Built by @wi11fr0sco</p>
        <p>Upload any image → get prediction</p>
        <form action="/predict" enctype="multipart/form-data" method="post" style="margin: 20px;">
            <input name="file" type="file" accept="image/*" style="padding: 10px;">
            <input type="submit" value="Predict" style="padding: 10px 20px; background: #1DA1F2; color: white; border: none; cursor: pointer;">
        </form>
        <p><em>Try your Tesla FSD image!</em></p>
    </div>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # preview image
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # predict
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = output.softmax(1)[0][pred_idx].item()
    
    # confidence bar
    bar_width = int(confidence * 300)
    bar_color = "#4CAF50" if confidence > 0.7 else "#FF9800" if confidence > 0.4 else "#F44336"
    
    return HTMLResponse(f"""
    <div style="font-family: Arial; text-align: center; margin: 40px;">
        <h2>Prediction: <strong>{classes[pred_idx].upper()}</strong></h2>
        <p>Confidence: <strong>{confidence:.1%}</strong></p>
        <div style="width: 300px; height: 20px; background: #ddd; margin: 20px auto; border-radius: 10px; overflow: hidden;">
            <div style="width: {bar_width}px; height: 100%; background: {bar_color}; transition: width 0.5s;"></div>
        </div>
        <img src="data:image/png;base64,{img_str}" style="max-width: 300px; margin: 20px; border: 2px solid #1DA1F2; border-radius: 10px;">
        <p><em>Built from scratch • Trained on CIFAR-10 • 79% accuracy</em></p>
        <a href="/" style="color: #1DA1F2;">← Try another</a>
    </div>
    """)

# run
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)