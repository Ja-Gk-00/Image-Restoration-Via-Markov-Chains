import cv2
import numpy as np
from pathlib import Path


from MarkovObjects.GibbsEstimator import (
    LorentzianLoss, SparseGradientPrior,
    GridMRF, GibbsSampler
)

def generate_and_save_denoised_images(image_path_str: str, output_folder_str: str):
    image_path = Path(image_path_str)
    output_folder = Path(output_folder_str)

    output_folder.mkdir(parents=True, exist_ok=True)

    loaded_noisy_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if loaded_noisy_image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    
    noisy = cv2.cvtColor(loaded_noisy_image, cv2.COLOR_BGR2RGB)
    
    noisy = cv2.resize(noisy, (400, 300), interpolation=cv2.INTER_AREA)
    
    downscale_factor = 0.5
    noisy = cv2.resize(
        noisy,
        (0, 0),
        fx=downscale_factor,
        fy=downscale_factor,
        interpolation=cv2.INTER_AREA
    )

    loss = LorentzianLoss(sigma=1)
    prior = SparseGradientPrior(p=1)
    mrf = GridMRF(noisy, loss, prior,
                  lambda_r=1, window_term=None, lambda_w=1)

    num_iter = 80
    burn_in = 40
    
    # Configuration 1: MAPE = SA + Quadratic
    betas_sa = np.linspace(0.01, 1, num_iter)
    sampler_mape_sa_quad = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='map', 
        pior_type_for_optimal='quadratic', 
        betas=betas_sa
    )
    sampler_mape_sa_quad.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mape_sa_quad = sampler_mape_sa_quad.estimate()
    output_file_mape_sa_quad = output_folder / f"{image_path.stem}_mape_sa_quadratic{image_path.suffix}"
    cv2.imwrite(str(output_file_mape_sa_quad), cv2.cvtColor(denoised_mape_sa_quad.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Configuration 2: MAPE = SA + Potts
    sampler_mape_sa_potts = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='map', 
        pior_type_for_optimal='potts', 
        betas=betas_sa
    )
    sampler_mape_sa_potts.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mape_sa_potts = sampler_mape_sa_potts.estimate()
    output_file_mape_sa_potts = output_folder / f"{image_path.stem}_mape_sa_potts{image_path.suffix}"
    cv2.imwrite(str(output_file_mape_sa_potts), cv2.cvtColor(denoised_mape_sa_potts.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Configuration 3: MMSE - quadratic
    sampler_mmse_quad = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='mean', 
        pior_type_for_optimal='quadratic'
    )
    sampler_mmse_quad.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mmse_quad = sampler_mmse_quad.estimate()
    output_file_mmse_quad = output_folder / f"{image_path.stem}_mmse_quadratic{image_path.suffix}"
    cv2.imwrite(str(output_file_mmse_quad), cv2.cvtColor(denoised_mmse_quad.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Configuration 4: MMSE - potts
    sampler_mmse_potts = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='mean', 
        pior_type_for_optimal='potts'
    )
    sampler_mmse_potts.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mmse_potts = sampler_mmse_potts.estimate()
    output_file_mmse_potts = output_folder / f"{image_path.stem}_mmse_potts{image_path.suffix}"
    cv2.imwrite(str(output_file_mmse_potts), cv2.cvtColor(denoised_mmse_potts.astype(np.uint8), cv2.COLOR_RGB2BGR))