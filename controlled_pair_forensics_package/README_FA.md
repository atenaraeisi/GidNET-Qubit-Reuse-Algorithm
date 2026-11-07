# تحلیل عمیق دو جفت reuse plan هم‌عرض — مرحله A

این نسخه تکمیل‌کنندهٔ `structural_forensics.py` قبلی است. هدفش **شناختن تفاوت‌های موضعی در ساختار مدار قبل از Routing** است، نه انتخاب بهترین plan یا ادعای پیدا کردن علت قطعی Physical Cost.

## مقایسه‌های پیش‌فرض

1. `system9.qasm`: **P5** در برابر **P4** (کم‌عمق‌تر در برابر پرعمق‌تر؛ عرض ۷)
2. `sym9_146.qasm`: **P8** در برابر **P4** (کم‌عمق‌تر در برابر پرعمق‌تر؛ عرض ۵)

این دو جفت عمداً انتخاب شدند چون normalized logical depth و max weighted degree هر جفت یکسان‌اند، اما routed depth یکسان نیست.

## اجرا در PowerShell

پوشهٔ `controlled_pair_forensics_package` را داخل ریشهٔ پروژهٔ GidNET کپی کنید؛ همان‌جا که پوشهٔ `GPT` شامل چهار CSV قرار دارد.

```powershell
.\.venv_gidnet\Scripts\Activate.ps1

python .\controlled_pair_forensics_package\controlled_pair_forensics.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --out-dir ".\controlled_pair_results"
```

مسیر QASM باید **مدارهای اصلیِ استفاده‌شده در آزمایش‌های قبلی** را داشته باشد. نام فایل‌ها `system9.qasm` و `sym9_146.qasm` است. در صورت نبود هر فایل، برنامه فقط اطلاعات plan، ویژگی‌های قبلی و نتایج ۳۰-seed را برای آن جفت گزارش می‌کند و صریحاً `MISSING_QASM__SUMMARY_ONLY` می‌نویسد؛ هیچ تحلیل gate-level ساختگی‌ای تولید نمی‌کند.

### تست سریع کد

```powershell
python -m unittest discover -s .\controlled_pair_forensics_package -p "test_controlled_pair_forensics.py" -v
```

### انتخاب جفت دلخواه

```powershell
python .\controlled_pair_forensics_package\controlled_pair_forensics.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --out-dir ".\my_pair_results" `
  --pair system9.qasm P5 P4 `
  --pair sym9_146.qasm P8 P4
```

## خروجی‌ها را چطور بخوانیم؟

فایل‌های اصلی:

| فایل | معنایش |
|---|---|
| `REPORT.md` | گزارش انسانی، از دادهٔ واقعی و وضعیت موجود بودن QASM هر جفت |
| `pair_summary.csv` | ۳۰ routing seed ذخیره‌شده، جفت plan، delta depth/SWAP، تعداد gateهای تغییرکرده |
| `changed_reuse_boundaries.csv` | edgeهایی که حذف یا جایگزین شدند |
| `aligned_reuse_chains.csv` | مقایسهٔ chainهای دو plan بعد از تطبیق نام chainها بر اساس overlap |
| `*__moved_logical_qubits.csv` | logical qubitهای تغییرمکان‌یافته، تعداد interactionها و partnerهایشان |
| `*__changed_two_qubit_chain_pairs.csv` | مشخصات اصلی تک‌تک گیت‌های دوکیوبیتی که اتصال بین دو chain عوض کرده‌اند |
| `*__operation_asap_or_critical_changes.csv` | گیت‌هایی که ASAP layer یا عضویت در مسیر بحرانی پیش از Routing در آنها متفاوت است |
| `*__quotient_pair_load_differences.csv` | تغییر تعداد گیت‌های دوکیوبیتی برای هر جفت chain همترازشده |
| `*__per_chain_sequences.csv` | ترتیب interaction-partnerها روی هر chain در دو plan |
| `*__original_index_window_load.csv` | تعداد interactionهای هر chain در ۸ بازهٔ مشترک از شماره‌عملیات مدار اصلی |
| `figures/` | نمودار تغییر ASAP layer، محل تغییر interactionها، تفاوت فشار در بازه‌های original-QASM |
| `audit.json` | اطلاعات تکمیل بودن بازسازی و خلاصهٔ عددی جهت بازبینی |

کد **قبل از ارائهٔ خروجی gate-level** دوباره گراف تعاملِ ساخته‌شده از QASM و reuse chainها را با سه معیار ذخیره‌شدهٔ Experiment 2 (`max_weighted_degree`, `weighted_degree_cv`, `quotient_unique_edges`) تطبیق می‌دهد. اگر تطابق نباشد اجرای آن مدار با خطا متوقف می‌شود.

## نکات روش‌شناختی

- شمارهٔ chain به‌تنهایی معنی فیزیکی ندارد؛ chainها با بیشترین تعداد logical qubit مشترک هم‌تراز می‌شوند. در موارد متقارن، بیش از یک همترازی می‌تواند به یک اندازه درست باشد.
- گیت‌های دوکیوبیتی با **شمارهٔ گیت در QASM اصلی** ردیابی می‌شوند؛ ترتیب یا شمارهٔ گیت در مدار Routed اینجا وجود ندارد.
- ASAP layer در این تحلیل فرض می‌کند مدت همهٔ گیت‌ها یک واحد است؛ این **زمان‌بندی فیزیکی و اثر سخت‌افزار نیست**.
- بازه‌های زمانی `original_index_window_load` بر حسب **ترتیب اصلی QASM** هستند، نه ساعت اجرای کوانتومی یا میزان SWAP همزمان.
- برای تفاوت میانگین Routed Depth از ۳۰ seed ذخیره‌شده استفاده می‌شود؛ interval گزارش‌شده توصیفی است چون این جفت‌ها بعد از دیدن نتایج routing انتخاب شده‌اند.
- این مرحله نه route جدید اجرا می‌کند، نه reuse plan جدید تولید می‌کند، نه semantic equivalence را اثبات می‌کند.
- اگر نشانهٔ موضعی مشخصی دیدیم، مرحلهٔ بعد بررسی مدارهای routed واقعی با seed ثابت و سپس تست آن فرضیه روی planهای دیگر است.

## محتوای بسته

- `controlled_pair_forensics.py` — کد جدید
- `structural_forensics.py` — کد مرحلهٔ قبل، فقط برای import و بازاستفاده از parser/audits
- `test_controlled_pair_forensics.py` — تست دود و رگرسیون ساده
- `pilot_results/` — **خروجی اجراشده در اینجا**. برای `system9` تحلیل gate-level کامل است. برای `sym9_146` فقط خلاصهٔ CSV است چون QASM اصلی این مدار در فایل‌های این محیط موجود نبود. پس از اجرای فرمان با `--qasm-dir` درست روی سیستم خودت، آن بخش هم کامل می‌شود.
