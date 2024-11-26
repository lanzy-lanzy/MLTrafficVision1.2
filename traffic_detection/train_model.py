from ultralytics import YOLO
import os
from pathlib import Path

def train_custom_model(
    data_yaml="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch_size=16,
    device="cpu",
    project="runs/train",
    name="traffic_model"
):
    """
    Train a custom YOLO model for traffic detection
    """
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # start with pretrained YOLOv8n model

    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            verbose=True,
            exist_ok=True,
            patience=50,  # Early stopping patience
            save=True,  # Save checkpoints
            save_period=10,  # Save every 10 epochs
            plots=True,  # Generate plots
        )
        
        print("Training completed successfully!")
        
        # Validate the model
        metrics = model.val()
        print("\nValidation Metrics:")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        
        # Export to ONNX format for deployment
        onnx_path = model.export(format="onnx")
        print(f"\nModel exported to ONNX format: {onnx_path}")
        
        return True, model.best  # Return success and path to best model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False, None

if __name__ == "__main__":
    # Create dataset directory structure
    dataset_dir = Path("dataset")
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Example YAML content for dataset configuration
    yaml_content = """
# Train/val/test sets
path: dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
  5: tricycle
  6: emergency_vehicle
"""
    
    # Write YAML file
    with open(dataset_dir / "data.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print("Starting training process...")
    success, best_model = train_custom_model(
        data_yaml=str(dataset_dir / "data.yaml"),
        epochs=100,
        device="cpu"  # Change to "cuda" if you have a GPU
    )
    
    if success:
        print(f"\nTraining completed! Best model saved at: {best_model}")
        print("\nNext steps:")
        print("1. Place your training images in dataset/images/train")
        print("2. Place your validation images in dataset/images/val")
        print("3. Place corresponding labels in dataset/labels/train and dataset/labels/val")
        print("4. Run this script again to start training")
    else:
        print("\nTraining failed. Please check the error messages above.")
