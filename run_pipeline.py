"""
Standalone runner for the Phase 1 data pipeline.
Generates shared_data.csv identical to running notebook.ipynb cells 1-10.
Run from the repo root: python run_pipeline.py
"""
import os
import re
import email
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR: pathlib.Path = pathlib.Path("enron_mail_20150507/maildir")

PERSONAL_ACCOUNT_PATTERN: re.Pattern = re.compile(r"^[a-z]+(-[a-z0-9]+)+$")
MIN_EMAILS: int = 200
TOP_N: int = 20

FORWARDED_MARKER: str = "-----Original Message-----"

SIGNATURE_TRIGGER_PATTERNS: list[re.Pattern] = [
    re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"),
    re.compile(r"Director|Manager|Vice President|VP|Analyst|President|Officer|Associate", re.IGNORECASE),
    re.compile(r"Enron|ECT|ENA|Corp", re.IGNORECASE),
]

URL_PATTERN: re.Pattern = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN: re.Pattern = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}")


def count_emails_per_author(data_dir: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not PERSONAL_ACCOUNT_PATTERN.match(entry.name):
            continue
        file_count: int = sum(len(files) for _, _, files in os.walk(str(entry)))
        counts[entry.name] = file_count
    return counts


def select_top_authors(counts: dict[str, int], min_emails: int, top_n: int) -> list[str]:
    eligible: dict[str, int] = {a: c for a, c in counts.items() if c >= min_emails}
    sorted_authors: list[str] = sorted(eligible, key=lambda a: eligible[a], reverse=True)
    return sorted_authors[:top_n]


def _open_path(file_path: pathlib.Path):
    if os.name == "nt":
        path_str = "\\\\?\\" + str(file_path.absolute()).replace("/", "\\")
    else:
        path_str = str(file_path)
    return open(path_str, "r", encoding="utf-8", errors="replace")


def parse_email_body(file_path: pathlib.Path) -> str | None:
    with _open_path(file_path) as fh:
        msg: email.message.Message = email.message_from_file(fh)
    payload: str | bytes | list = msg.get_payload()
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for part in payload:
            if part.get_content_type() == "text/plain":
                part_payload = part.get_payload()
                if isinstance(part_payload, str):
                    return part_payload
    return None


def strip_forwarded(body: str) -> str:
    idx: int = body.find(FORWARDED_MARKER)
    if idx == -1:
        return body
    return body[:idx]


def strip_signature(body: str) -> str:
    lines: list[str] = body.rstrip("\n").splitlines()
    tail: list[str] = lines[-5:]
    match_count: int = sum(
        1 for line in tail
        if any(pat.search(line) for pat in SIGNATURE_TRIGGER_PATTERNS)
    )
    if match_count < 2:
        return body
    while lines and any(pat.search(lines[-1]) for pat in SIGNATURE_TRIGGER_PATTERNS):
        lines.pop()
    return "\n".join(lines)


def normalise(body: str) -> str:
    body = URL_PATTERN.sub("URL_TOKEN", body)
    body = EMAIL_PATTERN.sub("EMAIL_TOKEN", body)
    body = body.lower()
    return body


def is_long_enough(body: str, min_words: int = 20) -> bool:
    return len(body.split()) >= min_words


def process_author_emails(author_name: str, author_dir: pathlib.Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for dirpath, _, filenames in os.walk(str(author_dir)):
        for filename in sorted(filenames):
            file_path: pathlib.Path = pathlib.Path(dirpath) / filename
            raw_body: str | None = parse_email_body(file_path)
            if raw_body is None:
                continue
            body: str = strip_forwarded(raw_body)
            body = strip_signature(body)
            body = normalise(body)
            if not is_long_enough(body):
                continue
            records.append({"author_label": author_name, "cleaned_body": body})
    return records


def assign_splits(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.30, stratify=df["author_label"], random_state=random_state,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=df.loc[temp_idx, "author_label"], random_state=random_state,
    )
    df_out: pd.DataFrame = df.copy()
    df_out["split"] = "train"
    df_out.loc[val_idx, "split"] = "val"
    df_out.loc[test_idx, "split"] = "test"
    return df_out


if __name__ == "__main__":
    print("Scanning authors...")
    raw_counts = count_emails_per_author(DATA_DIR)
    selected_authors = select_top_authors(raw_counts, MIN_EMAILS, TOP_N)
    print(f"Authors with >= {MIN_EMAILS} emails: {len([c for c in raw_counts.values() if c >= MIN_EMAILS])}")
    print(f"Selected top {TOP_N}: {selected_authors}\n")

    print("Processing emails...")
    all_records: list[dict[str, str]] = []
    email_id_counter: int = 0
    for author in selected_authors:
        author_dir = DATA_DIR / author
        author_records = process_author_emails(author, author_dir)
        for record in author_records:
            record["email_id"] = f"email_{email_id_counter:06d}"
            email_id_counter += 1
            all_records.append(record)
        print(f"  {author}: {len(author_records)} emails after filtering")
    print(f"\nTotal records: {len(all_records)}")

    df = pd.DataFrame(all_records, columns=["email_id", "author_label", "cleaned_body"])
    assert df["email_id"].is_unique
    assert df["author_label"].nunique() == TOP_N

    df_split = assign_splits(df, random_state=42)
    split_counts = df_split["split"].value_counts()
    total = len(df_split)
    print(f"\ntrain: {split_counts['train']} ({split_counts['train']/total:.1%})")
    print(f"val:   {split_counts['val']}   ({split_counts['val']/total:.1%})")
    print(f"test:  {split_counts['test']}  ({split_counts['test']/total:.1%})")

    output_path = pathlib.Path("shared_data.csv")
    df_split.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_split)} rows to {output_path.resolve()}")

    # Verification
    df_verify = pd.read_csv("shared_data.csv")
    assert list(df_verify.columns) == ["email_id", "author_label", "cleaned_body", "split"]
    assert df_verify["author_label"].nunique() == 20
    assert (df_verify["author_label"].value_counts() >= 200).all()
    for pattern in [r"^From:", r"^To:", r"^Subject:", r"^Date:", r"-----Original Message-----"]:
        assert not df_verify["cleaned_body"].str.contains(pattern, regex=True, na=False).any(), f"Leak: {pattern}"
    assert (df_verify["cleaned_body"].str.split().str.len() >= 20).all()
    train_pct = (df_verify["split"] == "train").sum() / total
    val_pct   = (df_verify["split"] == "val").sum()   / total
    test_pct  = (df_verify["split"] == "test").sum()  / total
    assert 0.68 <= train_pct <= 0.72
    assert 0.13 <= val_pct   <= 0.17
    assert 0.13 <= test_pct  <= 0.17

    print("\nAll Phase 1 verification checks passed.")
    print(f"  Authors: {df_verify['author_label'].nunique()}")
    print(f"  Total rows: {total}")
    print(f"  Train: {train_pct:.1%}  Val: {val_pct:.1%}  Test: {test_pct:.1%}")
