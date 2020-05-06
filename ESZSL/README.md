## Running the code

#### Train

```
python eszsl/eszsl_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -mode train -alpha [KERNEL SPACE REGULARIZER] -gamma [ATT SPACE REGULARIZER]
```
For testing, set mode to test and set alpha, gamma to best combination from tables below.

## Results

The numbers below are **class-averaged top-1 accuracies** (see ZSLGBU paper for details).

#### Classical ZSL

| Dataset | ZSLGBU Results| Repository Results | Hyperparams from Val |
|---------|:-------------:|:------------------:|:--------------------:|
| CUB     |     53.9      | 	   53.94 	   |Alpha=3, Gamma=-1     |
| AWA1    |   **58.2**    |        56.80       |Alpha=3, Gamma=0      |
| AWA2    |   **58.6**    |        54.82       |Alpha=3, Gamma=0      |
| aPY     |     38.3      |      **38.56**     |Alpha=3, Gamma=-1     |
| SUN     |     54.5      |      **55.69**     |Alpha=3, Gamma=2      |

#### Generalized ZSL

|Dataset || ZSLGBU Results      ||| Repository Results || Hyperparams from Val |
|--------|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|:--------------------:|
|        | U     | S     | H     | U     | S    | H     |            	       |
| CUB    | 12.6 | **63.8** | 21.0 | **14.70** | 56.53 | **23.34** |Alpha=3, Gamma=0 |
| AWA1   | **6.6** | 75.6 | **12.1** | 5.29 | **86.84** | 9.98 |Alpha=3, Gamma=0 |
| AWA2   | **5.9** | 77.8 | **11.0** | 4.04 | **88.84** | 7.72 |Alpha=3, Gamma=0 |
| aPY    | **2.4** | 70.1 | **4.6** | 2.25 | **81.07** | 4.39 |Alpha=2, Gamma=0 |
| SUN    | 11.0 | 27.9 | 15.8 | **13.75** | **28.41** | **18.53** |Alpha=3, Gamma=2 |

U -> Unseen Classes; S -> Seen Classes; H-> Harmonic Mean of the 2.

### References

[1] [Original MATLAB Code by Authors](https://github.com/bernard24/Embarrassingly-simple-ZSL)

[2] https://github.com/sbharadwajj/embarrassingly-simple-zero-shot-learning