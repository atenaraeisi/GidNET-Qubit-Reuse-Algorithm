# Experiment 4A — Physical Value of Expanded Neighborhood

**این بسته برای آزمایش 4A است؛ 4B و 4C عمداً داخل آن اجرا نمی‌شوند.**

## محل صحیح فایل‌ها در ویندوز

ریشهٔ پروژه:

`C:\Users\ASUS\Documents\Quantum computing\Qiskit\GidNET-Qubit-Reuse-Algorithm`

ساختار پوشه‌ها:

```text
GidNET-Qubit-Reuse-Algorithm\
├── experiment4A_physical_value.py      ← اسکریپت اصلی این بسته
├── Experiment4A_Run.ipynb             ← راهنمای اجرای سلولی (اختیاری)
├── gidnet\                           ← کد موجود پروژه (تغییر نده)
├── experiment3_targeted_hotspot_repair.ipynb (تغییر نده)
├── experiment2_results\
│   └── experiment2_plan_features.csv  ← فایل قبلی خودت
├── experiment4a_inputs\               ← خودت بساز
│   ├── adversarial_mutation_cases.csv  ← Full Coverage با 8642 mutation
│   └── width_and_original_plan_audit.csv
└── experiment4a_results\              ← خودش تولید می‌شود
```

ورودی `adversarial_mutation_cases.csv` باید **نسخهٔ کامل Full Coverage** (۸۶۴۲ ردیف) باشد، نه CSV قدیمیِ نمونه‌گیری‌شده. اگر نام فایل دانلودشده `(3)` دارد، **کپی آن را** با اسم استاندارد بالا در `experiment4a_inputs` بگذار. برای فایل Audit هم از `width_and_original_plan_audit(2).csv` نسخهٔ منطبق با همان Full Coverage استفاده کن، و اسم کپی را استاندارد کن.

پنج QASM اصلی همچنان در مسیر جداگانهٔ موجود در کامپیوترت هستند:

`C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks`

## اجرا در PowerShell

از ریشهٔ پروژه اجرا کن:

```powershell
cd "C:\Users\ASUS\Documents\Quantum computing\Qiskit\GidNET-Qubit-Reuse-Algorithm"

# ترجیحاً همان Python و نسخه‌های Qiskitِ آزمایش 3 را به کار ببر.
# اگر محیط مربوطه این فایل بود:
$PY = "C:\Users\ASUS\caqr-qiskit045-env\Scripts\python.exe"
& $PY -c "import qiskit; print(qiskit.__version__)"

# گام 1: بدون هیچ Routing؛ فقط ساخت کاتالوگ معتبر و چک اعداد
& $PY .\experiment4A_physical_value.py prepare

# گام 2: فقط یک transpile؛ بررسی سالم بودن کامپایل، بدون پایان آزمایش
& $PY .\experiment4A_physical_value.py explore --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" --smoke-routes 1

# گام 3: کامل کردن همهٔ Physical Evaluationهای فاز کشف؛ با امکان Resume
& $PY .\experiment4A_physical_value.py explore --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks"

# گام 4: آزمون مستقل روی Seedهای 100–129، بدون تغییر کاندیدای انتخاب‌شده
& $PY .\experiment4A_physical_value.py validate --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks"
```

**اگر Python این مسیر موجود نیست**، در دستورها به جای `& $PY` از Python فعالِ دقیقاً مربوط به آزمایش 3 استفاده کن. محیط `.venv_gidnet` قبلاً برای بعضی تست‌ها Qiskit 0.46.0 داشته؛ لزوماً همان محیط اجرای Routing آزمایش 3 نیست.

## خروجی‌ها

- `candidate_catalog.csv`، `source_memberships.csv`، `coverage_per_source.csv`: پوشش و حذف کاندیداهای تکراری.
- `inventory_report.json`: کنترل ۵۰ Plan، ۸۶۴۲ حرکت، ۴۸۵ Valid، و مجموع A=130، B=289، C=291.
- `routes_train.csv`: خروجی تک‌تک seedهای 42 تا 46؛ به محض تولید هر ردیف ذخیره می‌شود.
- `frozen_winners.csv`: برندگانِ جداگانهٔ SWAP/Depth برای A/B/C، با/بی محدودیت افزایش ۵٪ عمق منطقی.
- `opportunity_summary.csv`: فرصت اضافهٔ همسایگی گسترده در دادهٔ کشف.
- `routes_holdout.csv`: ارزیابی برندگان ثابت روی seedهای 100 تا 129.
- `heldout_summary.csv`، `heldout_by_circuit.csv`، `heldout_opportunity_comparison.csv`: نتایج فاز مستقل و تحلیل به تفکیک مدار.
- `run_manifest.json`: اثر انگشت ورودی‌ها و تنظیمات؛ اگر عوض شدند، اسکریپت اجازهٔ ترکیب با نتایج قبلی را نمی‌دهد.

## هشدار علمی مهم

Static Validity **معادل** تأیید کامل Measurement/Reset Semantics نیست. این اسکریپت در مرحلهٔ Routing از نسخهٔ patched GidNET استفاده می‌کند و chainهای ذخیره‌شده را جایگزین خروجی generator می‌کند. قبل از تفسیر علمی خروجی، بررسی کن که `GidNET.compile_to_dynamic_circuit` و `lower_reset_to_measure_x` دقیقاً با **pipeline واقعی آزمایش 3** سازگار باشند. اگر کد آزمایش سوم متفاوت است، می‌توانی با `--adapter FILE.py` یک bridge دقیق به آن بدهی. این اسکریپت ادعای تأیید semantic نمی‌کند؛ مستندات اعتبارسنجی جداگانه را نگه دار.

برای سنجش کیفیت Routing، SWAP و Depth از **همان کامپایل و Backend/Seed** استخراج می‌شوند. موفقیت در 4A تنها وجود «فرصت مشاهده‌شده» است، نه اثبات عملکرد 4B/4C یا مقایسهٔ بودجه‌ برابر با Experiment 3.

**دستورهای این README را لازم نیست در Jupyter اجرا کنی؛ می‌توانی مستقیم در PowerShell اجرا کنی.** فایل نوت‌بوک هم برای اجرای مرحله‌ای همراه بسته است.
