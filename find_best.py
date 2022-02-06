from predict import main as predict
from tqdm import tqdm
from fire import Fire


def main(model_path, start=0, end=10000, out='best.csv'):
    """
    python3 -u find_best.py --model_path=./models/20_2000000_22528.zip --out=best.csv --start=0 --end=10
    cat *.csv|awk -F ',' '{print $1"\t"$2}' | grep "0" | sort -n --reverse -k 2,2 | head -n 20
    """
    best = None
    best_score = None
    scores = []
    with open(out, 'w') as f:
        f.write('seed,score\n')
        pbar = tqdm(range(start, end))
        for i in pbar:
            r = predict(render=False, seed=i, model_path=model_path)
            scores.append({'seed': i, 'score': r})
            if best is None or r > best_score:
                best = i
                best_score = r
            pbar.set_description(f'best seed: {best}, score{best_score}')
            f.write(f'{i},{r}\n')
            f.flush()


if __name__ == '__main__':
    Fire(main)
