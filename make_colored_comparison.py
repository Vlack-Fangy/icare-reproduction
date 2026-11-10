"""Format the complete iCARE score comparison as an honest, color-coded XLSX.

The reproduced scores use original transcripts. A paired anonymized-transcript
summarization experiment has not been run, so this file does not claim one.
"""

from collections import defaultdict
from pathlib import Path
import csv

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/comparison.csv"
DEST = ROOT / "results/colored_full_comparison.xlsx"
METRICS = (
    ("bleu", "BLEU", True),
    ("meteor", "METEOR", True),
    ("rouge", "ROUGE-L", True),
    ("bertscore", "BERTScore", True),
    ("bleurt", "BLEURT", True),
    ("infolm", "InfoLM", False),
)
RED = PatternFill("solid", fgColor="F4CCCC")
BLUE = PatternFill("solid", fgColor="CFE2F3")
HEADER = PatternFill("solid", fgColor="213B54")
SUBHEADER = PatternFill("solid", fgColor="D9E5EF")


def load_rows():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 * 17 * len(METRICS):
        raise ValueError(f"Expected 204 comparison rows; got {len(rows)}")
    indexed = defaultdict(dict)
    for row in rows:
        key = (row["setting"], row["aspect"])
        if row["metric"] in indexed[key]:
            raise ValueError(f"Duplicate metric for {key}: {row['metric']}")
        indexed[key][row["metric"]] = row
    for key, metrics in indexed.items():
        if set(metrics) != {metric for metric, _, _ in METRICS}:
            raise ValueError(f"Incomplete metric set for {key}")
    return rows, indexed


def add_readme(workbook):
    sheet = workbook.active
    sheet.title = "Read Me"
    lines = [
        "iCARE Gemma 7B IT: full score comparison",
        "These reproduced scores are from summaries of ORIGINAL iCARE transcripts.",
        "No paired summarization scores exist yet for anonymized transcripts.",
        "Reported reference = the two three-decimal tables supplied by the user; their experiment captions are unconfirmed.",
        "Zero shot uses the first supplied table; one shot uses the second provisionally.",
        "Reproduction = this project's Gemma 7B run on 40 test sessions (41 expert-reference rows).",
        "Difference = reproduction minus reported reference. All 17 aspects and all 6 metrics appear on each setting sheet.",
        "RED = reproduction is better in the metric's direction; BLUE = equal or worse.",
        "For BLEU, METEOR, ROUGE-L, BERTScore and BLEURT, higher is better. For InfoLM, lower is better.",
        "The reported scores have only three decimal places; tiny differences and apparent gains may be rounding artifacts.",
        "These colors indicate differences from a reported table, not effects of anonymization or statistical significance.",
        "The anonymization batch currently has only four saved test candidates, all partial; no full paired test set is scored.",
    ]
    for number, line in enumerate(lines, 1):
        cell = sheet.cell(number, 1, line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if number == 1:
            cell.font = Font(bold=True, size=16, color="FFFFFF")
            cell.fill = HEADER
            sheet.row_dimensions[number].height = 31
        else:
            sheet.row_dimensions[number].height = 30
    sheet.column_dimensions["A"].width = 115


def add_setting(workbook, setting, indexed):
    sheet = workbook.create_sheet("Zero shot" if setting == "zeroshot" else "One shot")
    sheet.cell(1, 1, f"{sheet.title}: original transcripts vs user-supplied reference")
    sheet.cell(2, 1, "Red = better; blue = equal/worse. InfoLM improves when lower. No anonymized scores are in this sheet.")
    for row in (1, 2):
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=19)
        sheet.cell(row, 1).alignment = Alignment(wrap_text=True)
        sheet.row_dimensions[row].height = 29
    sheet.cell(1, 1).font = Font(bold=True, color="FFFFFF", size=14)
    sheet.cell(1, 1).fill = HEADER
    sheet.cell(3, 1, "Aspect")
    for number, (metric, label, _) in enumerate(METRICS):
        first = 2 + number * 3
        sheet.merge_cells(start_row=3, start_column=first, end_row=3, end_column=first + 2)
        sheet.cell(3, first, label)
        for offset, heading in enumerate(("Reported reference", "Reproduction", "Difference")):
            sheet.cell(4, first + offset, heading)
    for row in sheet.iter_rows(min_row=3, max_row=4, max_col=19):
        for cell in row:
            cell.fill = HEADER if cell.row == 3 else SUBHEADER
            cell.font = Font(bold=True, color="FFFFFF" if cell.row == 3 else "000000")
            cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    sheet.row_dimensions[3].height = 27
    sheet.row_dimensions[4].height = 34
    aspects = [aspect for candidate_setting, aspect in indexed if candidate_setting == setting]
    if len(aspects) != 17:
        raise ValueError(f"Expected 17 aspects for {setting}; got {len(aspects)}")
    for row_number, aspect in enumerate(aspects, 5):
        sheet.cell(row_number, 1, aspect)
        sheet.row_dimensions[row_number].height = 22
        for number, (metric, _, higher_is_better) in enumerate(METRICS):
            record = indexed[(setting, aspect)][metric]
            if record["status"] != "scored" or not record["ours"]:
                raise ValueError(f"Missing score: {setting}/{aspect}/{metric}")
            reference = float(record["reference"])
            ours = float(record["ours"])
            difference = ours - reference
            improved = difference > 0 if higher_is_better else difference < 0
            column = 2 + number * 3
            for offset, value in enumerate((reference, ours, difference)):
                cell = sheet.cell(row_number, column + offset, value)
                cell.number_format = '+0.000000;-0.000000;0.000000' if offset == 2 else '0.000000;[Black]-0.000000;0.000000'
                cell.alignment = Alignment(horizontal="right")
                if offset:
                    cell.fill = RED if improved else BLUE
    sheet.freeze_panes = "B5"
    sheet.auto_filter.ref = f"A4:S{4 + len(aspects)}"
    sheet.column_dimensions["A"].width = 38
    for column in range(2, 20):
        sheet.column_dimensions[get_column_letter(column)].width = 19
    sheet.sheet_view.zoomScale = 80


def main():
    _, indexed = load_rows()
    workbook = Workbook()
    add_readme(workbook)
    for setting in ("zeroshot", "oneshot"):
        add_setting(workbook, setting, indexed)
    workbook.save(DEST)
    print(DEST)


if __name__ == "__main__":
    main()
