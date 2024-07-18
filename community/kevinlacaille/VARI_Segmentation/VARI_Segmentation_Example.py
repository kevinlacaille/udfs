@fused.udf
def udf(bbox=None, resolution: int = 9, min_count: int = 10):

    from utils import get_bands, get_metadata, get_gsd, get_kernel_size, get_vari, threshold, smoothing, morphological_operations, segmentation

    image_path = 'https://github.com/kevinlacaille/presentations/blob/main/scipy2024/data/presentation/8928de8c247ffff/20496/DJI_0393.JPG'

    @fused.cache
    def read_data(image_path):
        # Get the bands of the image
        blue, green, red, rgb = get_bands(image_path)

    blue, green, red, rgb = read_data(image_path)

    # Get the metadata of the image
    metadata = get_metadata(image_path)

    # Get the GSD of the image
    gsd, gsd_tree = get_gsd(metadata)

    # Get the kernel size for morphological operations
    area_of_tree, kernel_size = get_kernel_size(gsd_tree)

    # Calculate the VARI index
    vari = get_vari(blue, green, red)

    # Generate the vegetation and non-vegetation masks
    vegetation_mask, non_vegetation_mask = threshold(vari)

    # Apply smoothing and morphological operations to the vegetation mask
    smoothed_mask = smoothing(vegetation_mask)

    # Apply morphological operations to the smoothed mask
    filtered_mask = morphological_operations(smoothed_mask, kernel_size)

    # Apply the mask to the RGB image
    mask_overlay, inverse_mask_overlay = segmentation(filtered_mask, rgb)

    return mask_overlay, inverse_mask_overlay
