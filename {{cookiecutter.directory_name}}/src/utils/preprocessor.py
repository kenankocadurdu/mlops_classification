import numpy
import cv2
import logging

class Executor:
    def __init__(self, image_size: int, remove_noise: bool, 
                 apply_clahe: bool, apply_gamma_corr: bool, median_filter: bool,
                 adaptive_mean_filter: bool, gaussian_filter: bool,
                 ) -> None:
        
        self.image_size = image_size
        self.remove_noise = remove_noise
        self.apply_clahe = apply_clahe
        self.apply_gamma_corr = apply_gamma_corr
        self.median_filter = median_filter
        self.adaptive_mean_filter = adaptive_mean_filter
        self.gaussian_filter = gaussian_filter

        self._log_initial_parameters()

    def _log_initial_parameters(self):
        logging.info("Executor Parameters:")
        params = vars(self)  # Fetches all class attributes
        for key, value in params.items():
            logging.info(f"{key}: {value}")

    def do_process(self, image: numpy.ndarray) -> numpy.ndarray:
        if self.remove_noise == True:
            image = remove_noise(image)
        if self.apply_clahe == True:
            image = apply_clahe(image) 
        if self.apply_gamma_corr == True:
            image = apply_gamma_corr(image)
        if self.median_filter == True:
            image = median_filter(image)
        if self.adaptive_mean_filter == True:
            image = adaptive_mean_filter(image)
        if self.gaussian_filter == True:
            image = gaussian_filter(image)
        
        image = resize_image(image, self.image_size)

        return image

def remove_noise(image: numpy.ndarray, ksize=3) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    output_image = image.copy()

    # Adjust brightness by decreasing intensity
    output_image = cv2.addWeighted(image, 1, output_image, -0.2, 0)

    # Apply Gaussian blur to reduce noise
    output_image = cv2.GaussianBlur(output_image, (ksize, ksize), 0)

    return output_image

def apply_clahe(image: numpy.ndarray, clip=2, tileGridSize=(32, 32)) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Create a CLAHE object with specified parameters
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tileGridSize)

    # Apply CLAHE to enhance the image
    output_image = clahe_create.apply(image)

    return output_image

def apply_gamma_corr(image: numpy.ndarray, gamma=1.3) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Normalize the image intensity values between 0 and 1
    normalized_image = image / 255.0

    # Apply gamma correction to the normalized image
    corrected_image = numpy.power(normalized_image, gamma) * 255.0

    # Rescale the corrected image intensity back to the range of 0-255 and convert to uint8
    output_image = corrected_image.astype(numpy.uint8)

    return output_image

def median_filter(image: numpy.ndarray, kernel_size=3) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Apply a median filter to the input image
    output_image = cv2.medianBlur(image, kernel_size)

    return output_image

def mean_filter(image: numpy.ndarray, kernel_size=(3, 3)) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Create a kernel matrix with ones and normalize it by the kernel size
    kernel = numpy.ones(kernel_size, numpy.float32) / (kernel_size[0] * kernel_size[1])

    # Apply the mean filter to the input image
    output_image = cv2.filter2D(image, -1, kernel)

    return output_image

def adaptive_mean_filter(image: numpy.ndarray, window_size=3, constant_c=15) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Create a grayscale copy of the input image
    gray_image = image.copy()

    # Get image dimensions
    rows, cols = gray_image.shape

    # Initialize an output image matrix
    output_image = numpy.zeros((rows, cols), dtype=numpy.uint8)

    # Calculate padding based on the window size
    pad = window_size // 2

    # Iterate over the image pixels
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            # Define the window bounds
            i_min = max(i - pad, 0)
            i_max = min(i + pad + 1, rows)
            j_min = max(j - pad, 0)
            j_max = min(j + pad + 1, cols)

            # Extract the local window
            window = gray_image[i_min:i_max, j_min:j_max]

            # Calculate the local mean within the window
            local_mean = numpy.mean(window)

            # Subtract the constant from the local mean and clip the result between 0 and 255
            output_image[i, j] = numpy.clip(local_mean - constant_c, 0, 255)

    return output_image

def gaussian_filter(image: numpy.ndarray, kernel_size=(5, 5), sigma_x=0) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Apply a Gaussian filter to the input image
    output_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)

    return output_image

def resize_image(image: numpy.ndarray, image_size: int = 256) -> numpy.ndarray:
    if image is None:
        raise ValueError("Could not read the image")

    # Resize the input image to the specified square image size
    output_image = cv2.resize(image, (image_size, image_size))

    return output_image


