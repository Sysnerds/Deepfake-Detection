from pathlib import Path
import pandas as pd

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Existing reports: reuse them, do not rerun the model
predictions_path = RESULTS_DIR / "external_evaluation_predictions.csv"
screening_path = RESULTS_DIR / "face_screening.csv"

predictions = pd.read_csv(predictions_path)
screening = pd.read_csv(screening_path)

# Normalize paths so the two reports can be matched reliably
predictions["match_path"] = (
    predictions["path"].astype(str).str.replace("\\", "/", regex=False).str.lower()
)
screening["match_path"] = (
    screening["image_path"].astype(str).str.replace("\\", "/", regex=False).str.lower()
)

# Match predictions with the face-screening results
merged = predictions.merge(
    screening[["match_path", "face_status", "face_count"]],
    on="match_path",
    how="left",
    validate="one_to_one",
    indicator=True,
)

# Verify that every prediction found its screening record
unmatched = int((merged["_merge"] != "both").sum())
print("===== REPORT MATCHING =====")
print(f"Prediction rows: {len(predictions)}")
print(f"Screening rows: {len(screening)}")
print(f"Unmatched predictions: {unmatched}")

if unmatched:
    print("\nSome paths did not match. Check the path formats before interpreting results.")
    print(merged.loc[merged["_merge"] != "both", ["path", "match_path"]].head(10))
    raise SystemExit("Analysis stopped because some image records could not be matched.")

# Summarize prediction behavior by screening result
print("\n===== RESULTS BY FACE-SCREENING GROUP =====")

for status, group in merged.groupby("face_status", dropna=False):
    print(f"\nGroup: {status}")
    print(f"Images: {len(group)}")
    print(f"Accuracy: {group['correct'].astype(str).str.upper().eq('TRUE').mean():.4f}")

    print("\nActual labels:")
    print(group["true_label"].value_counts().to_string())

    print("\nPredicted labels:")
    print(group["predicted_label"].value_counts().to_string())

# Save a separate analysis file without changing the original reports
output_path = RESULTS_DIR / "external_face_analysis.csv"
merged.drop(columns=["match_path", "_merge"]).to_csv(output_path, index=False)

print(f"\nDetailed analysis saved to: {output_path}")