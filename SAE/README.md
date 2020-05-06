## Running the code

#### Train

```
python sae/sae_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -mode train -ld1 [LOWER BOUND OF VARIATION] -ld2 [UPPER BOUND OF VARIATION]
```
For testing, set mode to test and set ld1 (F->S) and ld2 (S->F) to the best values from the tables below.

## Results

The numbers below are **class-averaged top-1 accuracies** (see ZSLGBU paper for details).

#### Classical ZSL

| Dataset | ZSLGBU Results || Repository Results                    |||
|---------|:--------------:|:--------:|:------:|:----------:|:-------:|
|         |                | F->S (W) | Lambda | S->F (W.T) | Lambda  |
| CUB     | 33.3           | 39.48    | 100    | **46.70**  | 0.2     |
| AWA1    | 53.0           | 51.34    | 3.0    | **59.89**  | 0.8     |
| AWA2    | 54.1           | 51.66    | 0.6    | **60.51**  | 0.2     |
| aPY     | 8.3            | 16.07    | 2.0    | **16.50**  | 4.0     |
| SUN     | 40.3           | 52.85    | 0.32   | **59.86**  | 0.16    |

#### Generalized ZSL

|Dataset ||ZSLGBU Results       |||||Repository Results      |||||
|--------|:-----:|:-----:|:-----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|        |       |       |       ||F->S (W)  || Lambda ||S->F (W.T)  || Lambda  |
|        | U     | S     | H     | U | S | H  |        | U | S | H    |         |
| CUB    | 7.8 | 54.0 | 13.6 | 13.86 | 49.88 | 21.69 | 80 | **15.72** | **57.02** | **24.64** | 0.2 |
| AWA1   | 1.8 | 77.1 | 3.5 | 5.29 | 80.52 | 9.92 | 3.2 | **14.72** | **82.93** | **25.0** | 0.8 |
| AWA2   | 1.1 | 82.2 | 2.2 | 5.0 | 81.42 | 9.42 | 0.8 | **12.86** | **87.20** | **22.41** | 0.2 |
| aPY    | 0.4 | **80.9** | 0.9 | 8.28 | 27.97 | 12.77 | 0.16 | **9.48** | 56.62 | **16.24** | 2.56 |
| SUN    | 8.8 | 18.0 | 11.8 | 16.81 | 24.69 | 20.0 | 0.32 | **19.03** | **31.20** | **23.64** | 0.08 |

U -> Unseen Classes; S -> Seen Classes; H-> Harmonic Mean of the 2.

### References

[1] [Original MATLAB Code by Authors](https://github.com/Elyorcv/SAE)

[2] https://github.com/hoseong-kim/sae-pytorch