from preprocessing.removeColumns import remove_columns
from preprocessing.activity_col import keep_required_columns


def main():
    remove_columns()
    keep_required_columns()  


if __name__ == "__main__":
    main()
