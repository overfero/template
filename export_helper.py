import argparse
import os
import torch
import shutil
from ultralytics import YOLO

def check_env():
    print("\n--- Environment Info ---")
    print(f"Torch version: {torch.version.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        import tensorrt
        print(f"TensorRT version: {tensorrt.__version__}")
    except ImportError:
        print("TensorRT: NOT INSTALLED (Required for .engine export)")

def export_all(model_path):
    """
    Exports the model to multiple formats:
    1. ONNX (CPU)
    2. OpenVINO (Intel CPU)
    3. ONNX FP16 (GPU)
    4. TensorRT (NVIDIA GPU)
    5. TFLite (Android, FP16)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    check_env()

    print(f"\nStarting export for: {model_path}")
    base_name = os.path.splitext(model_path)[0]
    
    # Load model
    model = YOLO(model_path)

    # 1. CPU ONNX
    print("\n--- [1] Exporting to CPU ONNX ---")
    onnx_cpu_path = model.export(format="onnx", dynamic=True, simplify=True, device='cpu')
    final_onnx_cpu = f"{base_name}_cpu.onnx"
    if os.path.exists(onnx_cpu_path) and onnx_cpu_path != final_onnx_cpu:
        if os.path.exists(final_onnx_cpu): os.remove(final_onnx_cpu)
        shutil.move(onnx_cpu_path, final_onnx_cpu)
    print(f"Saved: {final_onnx_cpu}")

    # 2. CPU Intel --> OpenVINO
    print("\n--- [2] Exporting to OpenVINO (Intel CPU) ---")
    model.export(format="openvino", device='cpu')

    print("\nAll requested formats processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model Export Helper")
    parser.add_argument("model", type=str, help="Path to the .pt model file")
    args = parser.parse_args()
    
    export_all(args.model)
