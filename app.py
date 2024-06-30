from flask import Flask, request, render_template, send_from_directory
import numpy as np
import cv2
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def add_noise(image, noise_type, param):
    if noise_type == 'gaussian':
        mean = 0
        var = param
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image + gauss
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = param
        noisy = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1]] = 1

        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1]] = 0
    elif noise_type == 'speckle':
        gauss = np.random.randn(*image.shape)
        noisy = image + image * gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def denoise_image(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Konversi file .tif ke .png untuk ditampilkan di web
        image = Image.open(filename)
        png_filename = os.path.splitext(filename)[0] + '.png'
        image.save(png_filename)
        
        # Baca gambar dalam format yang dapat diproses oleh OpenCV
        image = cv2.imread(png_filename, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 'Failed to load image'
        
        noisy_images = []
        psnr_values = []

        noise_type = request.form.get('noise_type')
        params = [float(p) for p in request.form.get('params').split(',')]
        
        for param in params:
            noisy_image = add_noise(image, noise_type, param)
            noisy_filename = f'noisy_{param}.png'
            noisy_file_path = os.path.join(app.config['UPLOAD_FOLDER'], noisy_filename)
            cv2.imwrite(noisy_file_path, noisy_image)
            noisy_images.append(noisy_filename)
            psnr_values.append(psnr(image, noisy_image))

        denoised_images = []
        denoised_psnr_values = []
        
        for noisy_filename in noisy_images:
            noisy_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], noisy_filename), cv2.IMREAD_GRAYSCALE)
            if noisy_image is None:
                return 'Failed to load noisy image'
            denoised_image = denoise_image(noisy_image)
            denoised_filename = f'denoised_{noisy_filename}'
            denoised_file_path = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)
            cv2.imwrite(denoised_file_path, denoised_image)
            denoised_images.append(denoised_filename)
            denoised_psnr_values.append(psnr(image, denoised_image))

        return render_template('index.html', filename=os.path.basename(png_filename), noisy_images=noisy_images,
                               psnr_values=psnr_values, denoised_images=denoised_images,
                               denoised_psnr_values=denoised_psnr_values, zip=zip)

if __name__ == "__main__":
    app.run(debug=True)
