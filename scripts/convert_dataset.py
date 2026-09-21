# Convert the .txt files in the data folder to a single csv file, making sure the quranic arabic and the translation are aligned by verse number.

# path to the Translation: data\quran_data\raw_data\en.sahih.txt
# path to the quranic arabic: data\quran_data\raw_data\quran-simple.txt


# head for en.sahih.txt:
    # 1|1|In the name of Allah, the Entirely Merciful, the Especially Merciful.
    # 1|2|[All] praise is [due] to Allah, Lord of the worlds -
    # 1|3|The Entirely Merciful, the Especially Merciful,
    # 1|4|Sovereign of the Day of Recompense.
    # 1|5|It is You we worship and You we ask for help.

# head for quran-simple.txt:
    # 1|1|بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ
    # 1|2|الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
    # 1|3|الرَّحْمَـٰنِ الرَّحِيمِ
    # 1|4|مَالِكِ يَوْمِ الدِّينِ
    # 1|5|إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ



import csv
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "quran_data"


def read_verses(path, text_column):
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["surah", "verse", text_column],
        comment="#",
        skip_blank_lines=True,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype={"surah": "Int64", "verse": "Int64", text_column: "string"},
        engine="python",
    )


def convert_dataset():
    translation_path = DATA_DIR / "raw_data" / "en.sahih.txt"
    arabic_path = DATA_DIR / "raw_data" / "quran-simple.txt"

    try:
        translation_df = read_verses(translation_path, "translation")
        arabic_df = read_verses(arabic_path, "arabic")
    except FileNotFoundError as error:
        print(f"File not found: {error.filename}")
        return 1
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as error:
        print(f"Error reading dataset: {error}")
        return 1

    key_columns = ["surah", "verse"]
    for name, dataframe in (("translation", translation_df), ("Arabic", arabic_df)):
        if dataframe[key_columns].isna().any().any():
            print(f"Missing surah or verse key in {name} dataset")
            return 1
        if dataframe.duplicated(key_columns).any():
            print(f"Duplicate surah/verse key in {name} dataset")
            return 1

    try:
        merged_df = pd.merge(
            arabic_df,
            translation_df,
            on=key_columns,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    except pd.errors.MergeError as error:
        print(f"Could not align datasets: {error}")
        return 1

    unmatched = merged_df[merged_df["_merge"] != "both"]
    if not unmatched.empty:
        print("Mismatch in surah/verse keys between the two datasets")
        print(unmatched[key_columns + ["_merge"]].to_string(index=False))
        return 1

    output_path = DATA_DIR / "merged_quran.csv"
    merged_df.drop(columns="_merge").to_csv(output_path, index=False, encoding="utf-8")
    print(f"Merged dataset saved to {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(convert_dataset())