from predict import main as predict


def main():
    best = None
    best_score = None
    for i in range(1000000):
        r = predict(render=False, seed=i)
        print(i, r, best, best_score)
        if best is None or r > best_score:
            best = i
            best_score = r


if __name__ == '__main__':
    main()
