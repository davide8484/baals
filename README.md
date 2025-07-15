This repository implements the algorithms in 

David Evans, Nan Ye. Blockwise acceleration of alternating least squares for canonical tensor decomposition. Numer Linear Algebra Appl. 2023;30(6):e2516. https://doi.org/10.1002/nla.2516. [PDF](https://onlinelibrary.wiley.com/doi/epdf/10.1002/nla.2516)

## Requirements

* python = 3.8.20
* pip install -r requirements.txt

## Algorithms and usage examples

The algorithms implemented in decomp.py.

See `experiments.py` for usage examples with explanations.

## Running experiments

Run the decomposition algorithms BA-ALS, BM-ALS, BP-ALS, M-ALS, ALS, herALS, LS, ELS on a simulated tensor, and generate a convergence plot:

```bash
$ python experiments.py 
```

See the paper for the description of the algorithms. 
