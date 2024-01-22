import torch
import io
import base64
import os

def tensor_to_base64(tensor):
    # Convert the tensor to a byte array
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    # Encode the byte array to base64 string
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def main():
    # Generate a sample tensor
    tensor = torch.randn(1000, 768)  # Assuming the tensor is of shape (1000, 768)

    # Convert tensor to base64
    base64_str = tensor_to_base64(tensor)

    # Get the size of the original tensor by saving it to a file
    torch.save(tensor, "tensor.pt")
    tensor_size = os.path.getsize("tensor.pt")

    # Get the size of the base64 string
    base64_size = len(base64_str.encode('utf-8'))

    # Cleanup the saved tensor file
    os.remove("tensor.pt")

    # Print the results
    print(f"Original Tensor Size: {tensor_size / (1024 * 1024):.2f} MB")
    print(f"Base64 Encoded Size: {base64_size / (1024 * 1024):.2f} MB")
    print(f"Size Increase Factor: {base64_size / tensor_size:.2f}")

if __name__ == "__main__":
    main()

