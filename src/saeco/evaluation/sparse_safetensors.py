import torch
from safetensors.torch import save_file, load_file, safe_open
from pydantic import BaseModel


class TensorShape(BaseModel):
    shape: list[int]


def save_sparse_tensor(sparse_tensor, filename):
    # Get the shape and indices of the sparse tensor
    shape = tuple(sparse_tensor.shape)
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    # Save the shape, indices, and values to a file
    save_file(
        {
            "indices": indices,
            "values": values,
        },
        filename,
        metadata={"json_dump": TensorShape(shape=shape).model_dump_json()},
    )


def load_sparse_tensor(filename):
    # Load the shape, indices, and values from the file
    with safe_open(filename, framework="pt", device="cpu") as f:
        shape = TensorShape.model_validate_json(f.metadata()["json_dump"]).shape

    loaded_data = load_file(filename)
    # print(loaded_data.keys())
    indices = loaded_data["indices"]
    values = loaded_data["values"]
    # Create a sparse COO  tensor from the loaded data
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.coalesce()


def load_sparse_tensor_force_shape(filename, shape, document_index_adder):
    # Load the shape, indices, and values from the file
    with safe_open(filename, framework="pt", device="cpu") as f:
        load_shape = TensorShape.model_validate_json(f.metadata()["json_dump"]).shape
    assert shape[1:] == load_shape[1:]

    loaded_data = load_file(filename)
    print(loaded_data.keys())
    indices = loaded_data["indices"]
    assert indices.shape[1] == 3
    indices[:, 0] += document_index_adder
    values = loaded_data["values"]
    # Create a sparse COO  tensor from the loaded data
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.coalesce()
