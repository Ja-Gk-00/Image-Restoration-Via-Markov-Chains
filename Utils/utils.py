from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from pathlib import Path


from MarkovObjects.GibbsEstimator import (
    LorentzianLoss, SparseGradientPrior,
    GridMRF, GibbsSampler
)


def mape_sa_quadratic(output_folder: Path, image_path: Path, num_iter: int, burn_in: int, mrf: GridMRF, betas_sa: np.ndarray):
    print("[DEBUG] Starting MAPE with SA + Quadratic")
    sampler_mape_sa_quad = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='map', 
        pior_type_for_optimal='quadratic', 
        betas_T=betas_sa
    )
    sampler_mape_sa_quad.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mape_sa_quad = sampler_mape_sa_quad.estimate()
    output_file_mape_sa_quad = output_folder / f"{image_path.stem}_mape_sa_quadratic{image_path.suffix}"
    cv2.imwrite(str(output_file_mape_sa_quad), cv2.cvtColor(denoised_mape_sa_quad.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return output_file_mape_sa_quad
    
def mape_sa_potts(output_folder: Path, image_path: Path, num_iter: int, burn_in: int, mrf: GridMRF, betas_sa: np.ndarray):
    print("[DEBUG] Starting MAPE with SA + Potts")
    sampler_mape_sa_potts = GibbsSampler(
        mrf, 
        num_iter=num_iter, 
        burn_in=burn_in, 
        verbose=False, 
        estimate_mode='map', 
        pior_type_for_optimal='potts', 
        betas_T=betas_sa
    )
    sampler_mape_sa_potts.fit_optimised(parallel_channels=True, shuffle_pixels=True)
    denoised_mape_sa_potts = sampler_mape_sa_potts.estimate()
    output_file_mape_sa_potts = output_folder / f"{image_path.stem}_mape_sa_potts{image_path.suffix}"
    cv2.imwrite(str(output_file_mape_sa_potts), cv2.cvtColor(denoised_mape_sa_potts.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return output_file_mape_sa_potts
    
def mmse_quadratic(output_folder: Path, image_path: Path, num_iter: int, burn_in: int, mrf: GridMRF):
    print("[DEBUG] Starting MMSE with Quadratic")
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
    return output_file_mmse_quad

def mmse_potts(output_folder: Path, image_path: Path, num_iter: int, burn_in: int, mrf: GridMRF):
    print("[DEBUG] Starting MMSE with Potts")
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
    return output_file_mmse_potts

def generate_and_save_denoised_images(image_path_str: str, output_folder_str: str):
    image_path = Path(image_path_str)
    output_folder = Path(output_folder_str)

    output_folder.mkdir(parents=True, exist_ok=True)

    loaded_noisy_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if loaded_noisy_image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    
    noisy = cv2.cvtColor(loaded_noisy_image, cv2.COLOR_BGR2RGB)
    

    loss = LorentzianLoss(sigma=1)
    prior = SparseGradientPrior(p=1)
    mrf = GridMRF(noisy, loss, prior,
                  lambda_r=1, window_term=None, lambda_w=1)

    num_iter = 80
    burn_in = 40
    betas_sa = np.linspace(0.01, 1, num_iter)
    
    
    # Define the tasks to run in parallel
    tasks = [
        ("mape_sa_quad", mape_sa_quadratic, (output_folder, image_path, num_iter, burn_in, mrf, betas_sa)),
        ("mape_sa_potts", mape_sa_potts, (output_folder, image_path, num_iter, burn_in, mrf, betas_sa)),
        ("mmse_quad", mmse_quadratic, (output_folder, image_path, num_iter, burn_in, mrf)),
        ("mmse_potts", mmse_potts, (output_folder, image_path, num_iter, burn_in, mrf))
    ]
    
    results = {}
    
    # Method 1: Using ProcessPoolExecutor (recommended)
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(task_func, *task_args): task_name 
            for task_name, task_func, task_args in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                results[task_name] = result
                print(f"✓ Completed: {task_name}")
            except Exception as e:
                print(f"✗ Error in {task_name}: {str(e)}")
                results[task_name] = None
    
    return results
    
    # # Configuration 1: MAPE = SA + Quadratic
    # output_file_mape_sa_quad = mape_sa_quadratic(output_folder, image_path, num_iter, burn_in, mrf, betas_sa)

    # # Configuration 2: MAPE = SA + Potts
    # output_file_mape_sa_potts = mape_sa_potts(output_folder, image_path, num_iter, burn_in, mrf, betas_sa)

    # # Configuration 3: MMSE - quadratic
    # output_file_mmse_quad = mmse_quadratic(output_folder, image_path, num_iter, burn_in, mrf)

    # # Configuration 4: MMSE - potts
    # output_file_mmse_potts = mmse_potts(output_folder, image_path, num_iter, burn_in, mrf)
    
    # return {
    #     "mape_sa_quad": output_file_mape_sa_quad,
    #     "mape_sa_potts": output_file_mape_sa_potts,
    #     "mmse_quad": output_file_mmse_quad,
    #     "mmse_potts": output_file_mmse_potts
    # }