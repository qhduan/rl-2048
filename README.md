2048 Python & Reinforcement Learning
===========

## Test result

Install dependencies

```bash
pip3 install r requirements.txt
```

Download [pretrained-model from GitHub release](https://github.com/qhduan/rl-2048/releases/tag/model)

Run exists model


```bash
python3 predict.py --model_path models/20_2000000_22528.zip --seed=323413
```

Best seed:

```csv
323413	127078.0
317843	124948.0
56284	124778.0
222476	124188.0
327475	124058.0
251098	124016.0
345447	124002.0
142442	123936.0
146288	123898.0
15770	123878.0
15823	123778.0
353089	123768.0
303930	123674.0
358024	123642.0
336750	123628.0
63017	123626.0
112921	123570.0
34155	123564.0
324249	123554.0
164055	123512.0
315851	123510.0
58060	123440.0
16765	123398.0
107331	123388.0
154929	123358.0
109963	123348.0
306593	123340.0
353438	123332.0
261145	123328.0
347378	123312.0
343945	123298.0
53061	123284.0
158862	123280.0
122007	123272.0
134632	123254.0
157997	123230.0
103111	123228.0
157027	123216.0
259223	123198.0
134644	123180.0
3028	123148.0
```

## Train

```bash
python3 train.py
```

## Readme original:

[![Run on Repl.it](https://repl.it/badge/github/yangshun/2048-python)](https://repl.it/github/yangshun/2048-python)

---

**⚠️NOTE⚠️**: We won't be accepting any contributions/changes to the project anymore. It is now readonly.

---

Based on the popular game [2048](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli. The game's objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048. Here is a Python version that uses TKinter! 

![screenshot](img/screenshot.png)

To start the game, run:
    
    $ python3 puzzle.py


Contributors:
==

- [Yanghun Tay](http://github.com/yangshun)
- [Emmanuel Goh](http://github.com/emman27)
