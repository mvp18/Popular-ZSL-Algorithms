## Running the code

#### Train

```
python ale/ale_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -e [EPOCHS] -es [EARLY STOP] -norm [NORMALIZATION TYPE] -lr [LEARNING RATE]
```
For testing, set learning rate (lr) and normalization type (norm) to best combination from the tables below.

## Results

The numbers below are **class-averaged top-1 accuracies** (see ZSLGBU paper for details).

#### Classical ZSL

| Dataset | ZSLGBU Results| Repository Results | Hyperparams from Val |
|---------|:-------------:|:------------------:|:--------------------:|
| CUB     |   **54.9**    | 	   44.43 	   |lr=1.0, norm=None     |
| AWA1    |   **59.9**    |        54.58       |lr=0.01, norm=L2      |
| AWA2    |   **62.5**    |        53.31       |lr=0.001, norm=L2     |
| aPY     |   **39.7**    |        33.21       |lr=0.002, norm=std    |
| SUN     |     58.1      |      **59.10**     |lr=0.1, norm=L2       |

#### Generalized ZSL

To be updated soon...

### References

No existing codebases found!