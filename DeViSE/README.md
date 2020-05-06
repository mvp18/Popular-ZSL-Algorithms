## Running the code

#### Train

```
python devise/devise_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -e [EPOCHS] -es [EARLY STOP] -norm [NORMALIZATION TYPE] -lr [LEARNING RATE] -mr [SVM LOSS MARGIN]
```
For testing, set learning rate (lr), margin (mr), and normalization type (norm) to best combination from the tables below.

## Results

The numbers below are **class-averaged top-1 accuracies** (see ZSLGBU paper for details).

#### Classical ZSL

| Dataset | ZSLGBU Results| Repository Results | Hyperparams from Val     |
|---------|:-------------:|:------------------:|:------------------------:|
| CUB     |   **52.0**    |      44.07         |lr=1.0, mr=1.0, norm=L2   |
| AWA1    |     54.2      |    **55.25**       |lr=0.01, mr=200, norm=std |
| AWA2    |   **59.7**    |      57.68         |lr=0.001, mr=150, norm=std|
| aPY     |   **39.8**    |      33.33         |lr=1.0, mr=1.0, norm=L2   |
| SUN     |   **56.5**    |      55.69         |lr=0.01, mr=3.0, norm=None|

#### Generalized ZSL

To be updated soon...

### References

No existing codebases found!