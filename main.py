from datasets import load_from_disk


def main():
    ds = load_from_disk(".wikipedia-dataset")
    for entry in ds.select([3,5,7]):
        print(entry)


if __name__ == "__main__":
    main()
