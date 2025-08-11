import h5py
import argparse
import os
import numpy as np
from PIL import Image


def visualize_hdf5(hdf5_path, save_dir):
    """Visualize HDF5 file contents and save as PNG"""
    os.makedirs(save_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as data:
        print(type(data))
        # Get frame number from filename
        frame_num = os.path.basename(hdf5_path).split('.')[0]
        
        # Process colors
        if 'colors' in data:
            colors = np.array(data['colors'])
            img = Image.fromarray((colors * 255).astype(np.uint8))
            img.save(os.path.join(save_dir, f'color_{frame_num}.png'))
        
        # Process depth (normalize for visualization)
        if 'depth' in data:
            depth = np.array(data['depth'])
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = Image.fromarray((depth_normalized * 255).astype(np.uint8))
            depth_img.save(os.path.join(save_dir, f'depth_{frame_num}.png'))

def convert_h5_to_hdf5(input_path, output_path=None):
    """
    Convert .h5 file to .hdf5 format (or reorganize data if needed)
    
    Args:
        input_path (str): Path to input .h5 file
        output_path (str): Path for output .hdf5 file (default: same as input with .hdf5 extension)
    """
    # Set default output path if not provided
    if output_path is None:
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}.hdf5"
    
    # Verify input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Converting {input_path} to {output_path}")
    
    # Open input and create output files
    with h5py.File(input_path, 'r') as h5_in, \
         h5py.File(output_path, 'w') as hdf5_out:
        
        # Copy all datasets and groups from input to output
        def copy_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Copy dataset
                hdf5_out.create_dataset(name, data=obj[()], compression=obj.compression)
                # Copy attributes
                for attr_name, attr_value in obj.attrs.items():
                    hdf5_out[name].attrs[attr_name] = attr_value
            elif isinstance(obj, h5py.Group):
                # Ensure group exists
                if name not in hdf5_out:
                    hdf5_out.create_group(name)
                # Copy group attributes
                for attr_name, attr_value in obj.attrs.items():
                    hdf5_out[name].attrs[attr_name] = attr_value
        
        # Process all items in input file
        h5_in.visititems(copy_items)
        
        print(f"Successfully converted {len(list(h5_in.keys()))} items")

def main():
    parser = argparse.ArgumentParser(description='Convert .h5 to .hdf5 format')
    parser.add_argument('input', help='Input .h5 file path')
    parser.add_argument('--output', '-o', help='Output .hdf5 file path (optional)')
    args = parser.parse_args()
    
    try:
        convert_h5_to_hdf5(args.input, args.output)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error during conversion: {e}")

    visualize_hdf5(args.output, '/home/devel/.draft/renderformer/scene_processor/output')


if __name__ == "__main__":
    main()