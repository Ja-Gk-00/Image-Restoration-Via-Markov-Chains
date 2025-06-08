# Image-Restoration-Via-Markov-Chains

![Image restoration with Bayesian Methods](denoising.png "Denoising with MCMC")

The repository consists of Markov Chains Monte Carlo (MCMC) methods to restore noisy or old photos.  
The two main methods used for restoration consist of applying:  
    - Simulated Annealing  
    - Gibbs Probing  
For each there are two loss functions implemented:  
    - Cut Quadratic loss  
    - Pott's model  

The entirety of the project's results and functionalities has been explored in the presentation (**NOTE** The presentation is fully in polish) linked in the project [Here](Presentation.pdf).  

Each having its own characteristics of image convertion (how it behaves, does it preserve contrast more (= Potts) or is sharper on the edges (= Cut Quadratic)).  
In more concrete terms:
### Gibbs Sampling

Gibbs sampling is a Markov chain Monte Carlo (MCMC) algorithm for generating samples from a multivariate probability distribution by iteratively sampling each variable from its conditional distribution given the current values of all other variables:  
    - At each iteration, every variable is updated in turn by sampling from its conditional distribution.  
    - Early samples (the “burn-in”) are discarded to allow the chain to converge toward the target distribution.  
    - After burn-in, the remaining samples form a collection that approximates the true posterior, enabling estimation of means, variances, or other statistics.  
    - Particularly useful when joint sampling is hard but conditional sampling is straightforward.  

### Simulated Annealing
Simulated annealing is a probabilistic meta-heuristic for global optimisation inspired by metallurgical annealing, which gradually lowers a material’s temperature to achieve a low-energy crystalline state:  
    - Begins with a high “temperature” that allows exploration of the solution space, including moves to worse solutions, to avoid early trapping in local minima.
    - The temperature is decreased according to a predetermined plan, gradually reducing randomness and focusing the search on promising regions.  
    - At each step, a candidate move is accepted with a probability that depends on the change in objective value and the current temperature, permitting uphill moves early on and favoring downhill moves as temperature falls.  
    - Theoretically guarantees convergence to a global optimum with an infinitely slow cooling schedule; in practice, a suitably chosen schedule yields high-quality solutions in finite time.  


## Installation
Follow the steps listed below to install the required packages and run denoising/restoration of the photos.  

### Setting up virtual environment + installing requirements
The user should set up a venv to install the requirements and run the script:  
    1. ``` python -m venv /path/to/virtual/env```,  
    2. ``` venv/bin/activate ```,  
    3. Then install requirements ```pip install -r requirements.txt ```

### Running the script
There is one main script which 4 prepared methods of denoising i.e: 
    1. MAPE + Simmulated Annealing + Cutoff Quadratic.  
    2. MAPE + Simmulated Annealing + Potts.  
    3. MMSE + Quadratic.  
    4. MMSE + Potts.  

To run the script one should run the following command from the main directory of the project:  
```\venv\bin\python.exe Scripts/run_all.py /path/to/clean/images /path/to/noisy/images /path/to/output ```.  
example:  
```python .\Scripts\run_all.py .\Data\Raw\examples\ .\Data\Transformed\examples\ .\Data\outptut\ ```.  

Where each argument represents:  
    1. */path/to/clean/images* - a path to clean images to which noised ones are available (should there be no clean images, input path to empty directory).  
    2. */path/to/noisy/images* - a path to directory that contains noised images.  
    3. */path/to/output* - where to output restored images and statistics about the restoration process.  

The clean images are only necessary to perform an evaluation of the denoising process that is later saved into a *.csv* file with metrics (e.g L1 and L2 measure).  
If they are not available, then the statistics will be generated as *NAN*.  
After running the script, for every noised image, four photos will be generated (each one denoised with a different method).  
Be mindfull that for large images, the process might take some time, even tough it is being processed by multitread for each channel of RGB.


### Contributors
The module and the presentation with theoretical background have been developed by two authors:  
    1. [github.com/Ja-Gk-00](https://github.com/Ja-Gk-00)  
    2. [github.com/trebacz626](https://github.com/trebacz626)  

The literature that the project was based on comes from the WUT lectures on Algorithmic Markov Chains Methods on the MINI faculty, as well as from the following sources:  
    1. Wong, A., Mishra, A., Zhang, W., Fieguth, P., & Clausi, D. A., “Stochastic image denoising based on Markov‐chain Monte Carlo sampling,” Signal Processing 91 (2011) 2112–2120  
    2. Yue, C., “Markov Random Fields and Gibbs Sampling for Image Denoising,” EE367 Report, Stanford University (2018)  
