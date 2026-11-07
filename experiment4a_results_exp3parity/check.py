import pandas as pd

f = pd.read_csv("frozen_winners.csv", dtype={"plan_id": str})

f["depth_cap"] = (
    f["depth_cap"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
)

# آیا محدودیت عمق، انتخاب نهایی را عوض کرده است؟
by_cap = f.pivot(
    index=["circuit", "plan_id", "objective", "pool"],
    columns="depth_cap",
    values="selected_key"
)

print("تعداد انتخاب‌های متفاوت با/بدون محدودیت:")
print((by_cap[True] != by_cap[False]).sum())

# آیا B و C دقیقاً یک طرح نهایی انتخاب کرده‌اند؟
bc = f[f["pool"].isin(["B_suffix_chain", "C_all"])].pivot(
    index=["circuit", "plan_id", "objective", "depth_cap"],
    columns="pool",
    values="selected_key"
)

print("تعداد انتخاب‌های متفاوت بین B و C:")
print((bc["B_suffix_chain"] != bc["C_all"]).sum())