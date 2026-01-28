from ultralytics import YOLO

# The library handles the "Class" for you
model = YOLO("atm.pt") 
model.export(format="onnx", opset=17)  # This does exactly what the manual script does