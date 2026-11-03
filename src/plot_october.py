import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("reports/october_predictions.csv")
df["date"] = pd.to_datetime(df["date"])

CAP = 6000
n_capped = (df["actual_rate"] > CAP).sum()

fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

mask = df["actual_rate"] <= CAP
ax.scatter(
    df.loc[mask, "actual_rate"],
    df.loc[mask, "predicted_rate"],
    alpha=0.25,
    s=6,
    color="#064A56",
    label=f"Shipments (n={mask.sum():,})",
)
ax.plot([0, CAP], [0, CAP], color="#E05A2B", linewidth=1.4, linestyle="--", label="Perfect prediction")

ax.set_xlabel("Actual rate ($)")
ax.set_ylabel("Predicted rate ($)")
ax.set_title("October 2025 - Actual vs Predicted Rate", loc="left", fontweight="bold", pad=10)
ax.set_xlim(0, CAP)
ax.set_ylim(0, CAP)
ax.legend(fontsize=9)
ax.grid(color="#E0E8EA", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

if n_capped > 0:
    ax.text(0.02, 0.96, f"{n_capped} shipments above ${CAP:,} not shown",
            transform=ax.transAxes, fontsize=8, color="#666", va="top")

fig.tight_layout()
fig.savefig("reports/october_actual_vs_predicted.png", bbox_inches="tight")
plt.close(fig)
print("Saved: reports/october_actual_vs_predicted.png")


daily = (
    df.groupby("date")[["actual_rate", "predicted_rate"]]
    .mean()
    .sort_index()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

ax.plot(daily["date"], daily["actual_rate"],
        color="#064A56", linewidth=2, marker="o", markersize=4, label="Actual (daily avg)")
ax.plot(daily["date"], daily["predicted_rate"],
        color="#E05A2B", linewidth=2, marker="o", markersize=4, linestyle="--", label="Predicted (daily avg)")

ax.set_title("October 2025 - Daily Average Rate: Actual vs Predicted", loc="left", fontweight="bold", pad=10)
ax.set_ylabel("Average rate ($)")
ax.legend()
ax.grid(axis="y", color="#E0E8EA", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=30)

fig.tight_layout()
fig.savefig("reports/october_daily_avg.png", bbox_inches="tight")
plt.close(fig)
print("Saved: reports/october_daily_avg.png")


errors = df["error"]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

ax.hist(errors.clip(-2000, 2000), bins=80, color="#064A56", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="#E05A2B", linewidth=1.5, linestyle="--", label="Zero error")
ax.axvline(errors.mean(), color="#F5A623", linewidth=1.5, linestyle=":", label=f"Mean error: ${errors.mean():.0f}")

ax.set_xlabel("Error = Actual - Predicted ($)  [clipped at +-$2,000]")
ax.set_ylabel("Count")
ax.set_title("October 2025 - Prediction Error Distribution", loc="left", fontweight="bold", pad=10)
ax.legend(fontsize=9)
ax.grid(axis="y", color="#E0E8EA", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("reports/october_error_distribution.png", bbox_inches="tight")
plt.close(fig)
print("Saved: reports/october_error_distribution.png")

print(f"\nMAE:  ${df['absolute_error'].mean():.2f}")
print(f"Median error: ${errors.median():.2f}")
print(f"% underpredicted: {(errors > 0).mean()*100:.1f}%")
print(f"% within $100:    {(df['absolute_error'] <= 100).mean()*100:.1f}%")
print(f"% within $250:    {(df['absolute_error'] <= 250).mean()*100:.1f}%")
