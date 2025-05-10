import csv
import os.path
from typing import List, Dict

import numpy as np
from PIL import Image
import pywt
from matplotlib import pyplot as plt
import pandas as pd


class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError
        img = np.array(Image.open(path), dtype=np.float32)
        return img

    @staticmethod
    def normalize(img: np.ndarray, dtype: str) -> np.ndarray:
        max_img = np.max(img)
        min_img = np.min(img)

        normalized_image = (img - min_img) / (max_img - min_img) * 255
        if dtype == 'uint8':
            return normalized_image.astype(np.uint8)
        else:
            return nomalized_image.astype(np.float32)

    @staticmethod
    def save_image(path: str, img: np.ndarray):
        if not img.dtype == np.uint8:
            ImageUtils.normalize(img, dtype='uint8')
        return Image.fromarray(img).save(path)


class WaveletUtils:
    @staticmethod
    def dwt2(image: np.ndarray, mother_wave: str) -> np.ndarray:
        coefficients = pywt.dwt2(image, mother_wave)
        return coefficients

    @staticmethod
    def idwt2(coefficients: tuple, mother_wave: str) -> np.ndarray:
        reconstructed_image = pywt.idwt2(coefficients, mother_wave)
        return reconstructed_image

    @staticmethod
    def wavedec2(image: np.ndarray, mother_wave: str, level: int) -> List[np.ndarray]:
        coefficients = pywt.wavedec2(image, mother_wave, level=level)
        return coefficients

    @staticmethod
    def waverec2(coefficients_multilevel: tuple, mother_wave: str) -> np.ndarray:
        reconstructed_multilevel = pywt.waverec2(coefficients_multilevel, mother_wave)
        return reconstructed_multilevel


class Calculations:
    @staticmethod
    def calculate_mse(a: np.ndarray, b: np.ndarray) -> float:
        c = (a - b) ** 2
        return np.mean(c)

    @staticmethod
    def thresholding(array: np.ndarray, threshold: float) -> np.ndarray:
        mask = np.where(array > threshold, 1, 0)
        return array * mask


class Helpers:
    @staticmethod
    def create_directories(directories: Dict[str, str]) -> None:
        for path in directories.values():
            os.makedirs(path, exist_ok=True)
        print(f"{BColors.OkGREEN}Directories created successfully.{BColors.ENDC}")
        return

    @staticmethod
    def plot_comparison(
            original_image: np.ndarray,
            processed_image: np.ndarray,
            title_original: str = "Original Image",
            title_processed: str = "Processed Image",
            save_path: str = None
    ) -> None:

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(title_original)
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(title_processed)
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)

        return plt.show()

    @staticmethod
    def save_results_to_csv(results: Dict[str, Dict[str, float]], save_path: str) -> None:
        wavelet_families = list(next(iter(results.values())).keys())
        header = ['Wavelet'] + list(results.keys())

        rows = []
        for wavelet in wavelet_families:
            row = [wavelet]
            for method in results.keys():
                row.append(results[method].get(wavelet, float('nan')))
            rows.append(row)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"{BColors.OkGREEN}Results saved to CSV file: {save_path}{BColors.ENDC}")
        return

    @staticmethod
    def plot_results_bar_chart(results: Dict[str, Dict[str, float]], save_path: str = None) -> None:
        data = {}
        for method, wavelets in results.items():
            method_name = method.replace('_mse', '').replace('_', ' ').title()
            data[method_name] = wavelets

        df = pd.DataFrame(data)

        ax = df.plot(kind='bar', figsize=(14, 8), width=0.8)
        plt.title('MSE Comparison Across Different Wavelet Families and Methods', fontsize=16)
        plt.xlabel('Wavelet Family', fontsize=14)
        plt.ylabel('Mean Square Error', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Method', fontsize=12)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{BColors.OkGREEN}Results chart saved to: {save_path}{BColors.ENDC}")

        return plt.show()


def main():
    directories = {
        "image_base_path": "Images",
        "image_thresholding_base_path": "Images/Thresholding",
        "image_1level_dwt_base_path": "Images/1-Level DWT-Zero",
        "image_2level_dwt_base_path": "Images/2-Level DWT-Zero",
        "results_path": "Results"
    }
    Helpers.create_directories(directories)

    image_path = os.path.join(directories["image_base_path"], "Fig0726(a).tif")
    original_image = ImageUtils.load_image(image_path)
    print(f"{BColors.HEADER}{BColors.BOLD}WAVELET IMAGE ANALYSIS{BColors.ENDC}")
    print(f"{BColors.OkCYAN}Loaded image shape: {original_image.shape}{BColors.ENDC}")

    wavelet_families = [
        'haar', 'db2', 'db4', 'sym2', 'sym4',
        'coif1', 'coif3', 'bior1.3', 'bior2.2', 'bior3.1'
    ]

    results = {
        "thresholding_mse": {},
        "1level_approximation_mse": {},
        "2level_approximation_mse": {}
    }

    for wavelet in wavelet_families:
        print(50 * "=")
        print(f"\n{BColors.HEADER}{BColors.BOLD}PROCESSING WAVELET: {wavelet}{BColors.ENDC}")

        # STAGE 1: Wavelet Transform with Thresholding
        print(f"\n{BColors.OkBLUE}{BColors.BOLD}STAGE 1: Wavelet Transform with Thresholding{BColors.ENDC}")

        wavelet_coefficients = WaveletUtils.dwt2(original_image, mother_wave=wavelet)
        cA, (cH, cV, cD) = wavelet_coefficients

        threshold_value = 0.92
        cH_thresholded = Calculations.thresholding(cH, threshold=threshold_value)
        cV_thresholded = Calculations.thresholding(cV, threshold=threshold_value)
        cD_thresholded = Calculations.thresholding(cD, threshold=threshold_value)

        thresholded_coefficients = cA, (cH_thresholded, cV_thresholded, cD_thresholded)
        thresholded_image = WaveletUtils.idwt2(thresholded_coefficients, mother_wave=wavelet)

        title_processed = f"Thresholded Reconstruction ({wavelet})"
        Helpers.plot_comparison(
            original_image,
            thresholded_image,
            title_processed=title_processed
        )

        mse = Calculations.calculate_mse(original_image, thresholded_image)
        results["thresholding_mse"][wavelet] = mse

        thresholded_path = os.path.join(
            directories["image_thresholding_base_path"],
            f"reconstructed_thresholded_{wavelet}.png"
        )
        ImageUtils.save_image(thresholded_path, ImageUtils.normalize(thresholded_image, dtype='uint8'))

        print(f"{BColors.OkGREEN}MSE: {mse:.4f} | Saved to: {thresholded_path}{BColors.ENDC}")

        # STAGE 2: 1-Level Wavelet Transform with Zero Detail Coefficients
        print(
            f"\n{BColors.OkBLUE}{BColors.BOLD}STAGE 2: 1-Level Wavelet Transform with Zero Detail Coefficients{BColors.ENDC}")

        zero_H = np.zeros_like(cH)
        zero_V = np.zeros_like(cV)
        zero_D = np.zeros_like(cD)

        approximation_only_coefficients = cA, (zero_H, zero_V, zero_D)
        approximation_only_image = WaveletUtils.idwt2(
            approximation_only_coefficients,
            mother_wave=wavelet
        )

        title_processed = f"1-Level Approximation Only ({wavelet})"
        Helpers.plot_comparison(
            original_image,
            approximation_only_image,
            title_processed=title_processed
        )

        mse = Calculations.calculate_mse(original_image, approximation_only_image)
        results["1level_approximation_mse"][wavelet] = mse

        approximation_path = os.path.join(
            directories["image_1level_dwt_base_path"],
            f"reconstructed_approximation_{wavelet}.png"
        )
        ImageUtils.save_image(approximation_path, ImageUtils.normalize(approximation_only_image, dtype='uint8'))

        print(f"{BColors.OkGREEN}MSE: {mse:.4f} | Saved to: {approximation_path}{BColors.ENDC}")

        # STAGE 3: 2-Level Wavelet Transform with Zero Detail Coefficients
        print(
            f"\n{BColors.OkBLUE}{BColors.BOLD}STAGE 3: 2-Level Wavelet Transform with Zero Detail Coefficients{BColors.ENDC}")

        wavelet_coefficients_2level = WaveletUtils.wavedec2(
            original_image,
            mother_wave=wavelet,
            level=2
        )
        new_coeffs = [wavelet_coefficients_2level[0]]

        level_2_zeros = tuple(np.zeros_like(detail) for detail in wavelet_coefficients_2level[1])
        new_coeffs.append(level_2_zeros)

        level_1_zeros = tuple(np.zeros_like(detail) for detail in wavelet_coefficients_2level[2])
        new_coeffs.append(level_1_zeros)

        approximation_only_image_2level = WaveletUtils.waverec2(
            new_coeffs,
            mother_wave=wavelet
        )

        title_processed = f"2-Level Approximation Only ({wavelet})"
        Helpers.plot_comparison(
            original_image,
            approximation_only_image_2level,
            title_processed=title_processed
        )

        mse = Calculations.calculate_mse(original_image, approximation_only_image_2level)
        results["2level_approximation_mse"][wavelet] = mse

        approximation_path_2level = os.path.join(
            directories["image_2level_dwt_base_path"],
            f"reconstructed_2level_approximation_{wavelet}.png"
        )
        ImageUtils.save_image(approximation_path_2level,
                              ImageUtils.normalize(approximation_only_image_2level, dtype='uint8'))

        print(f"{BColors.OkGREEN}MSE: {mse:.4f} | Saved to: {approximation_path_2level}{BColors.ENDC}")

    # Print summary of results
    print(f"\n{BColors.HEADER}{BColors.BOLD}SUMMARY OF RESULTS (MSE VALUES){BColors.ENDC}")

    header = f"{BColors.UNDERLINE}{'Wavelet':10} | {'Thresholding':15} | {'1-Level Approx':15} | {'2-Level Approx':15}{BColors.ENDC}"
    print(header)

    for wavelet in wavelet_families:
        thresh_mse = results["thresholding_mse"].get(wavelet, float('nan'))
        level1_mse = results["1level_approximation_mse"].get(wavelet, float('nan'))
        level2_mse = results["2level_approximation_mse"].get(wavelet, float('nan'))

        print(
            f"{BColors.OkCYAN}{wavelet:10} | {thresh_mse:15.4f} | {level1_mse:15.4f} | {level2_mse:15.4f}{BColors.ENDC}")

    # Save results to CSV and generate chart
    csv_path = os.path.join(directories["results_path"], "wavelet_analysis_results.csv")
    Helpers.save_results_to_csv(results, csv_path)

    chart_path = os.path.join(directories["results_path"], "wavelet_analysis_chart.png")
    Helpers.plot_results_bar_chart(results, chart_path)


if __name__ == '__main__':
    main()