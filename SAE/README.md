## Running the code

#### Train

```
python sae/sae_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -mode train
```
For testing, set mode to test and set alpha, gamma to best combination from tables below.

## Results

#### Classical ZSL

| Dataset | ZSLGBU Results | Respository Results                   ||||
|---------|:--------------:|:----------------------------------------:|
|         |                | F->S (W) | Lambda | S->F (W^T) | Lambda  |
| CUB     | 33.3           | 39.48    | 100    | **46.70**  | 0.2     |
| AWA1    | 53.0           | 51.34    | 3.0    | **59.89**  | 0.8     |
| AWA2    | 54.1           | 51.66    | 0.6    | **60.51**  | 0.2     |
| aPY     | 8.3            | 16.07    | 2.0    | **16.50**  | 4.0     |
| SUN     | 40.3           | 52.85    | 0.32   | **59.86**  | 0.16    |

#### Generalized ZSL

|Dataset |ZSLGBU Results       |||Respository Results     ||||||||
|--------|:-----:|:-----:|:-----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|        |       |       |       | F->S (W) ||| Lambda | S->F (W^T) ||| Lambda  |
|        | U     | S     | H     | U | S | H  |        | U | S | H    |         |
| CUB    | 12.6 | **63.8** | 21.0 | **14.70** | 56.53 | **23.34** | 1 | **14.70** | 56.53 | **23.34** | 1 |

### References

[1] [Original MATLAB Code by Authors](https://github.com/Elyorcv/SAE)

[2] https://github.com/hoseong-kim/sae-pytorch