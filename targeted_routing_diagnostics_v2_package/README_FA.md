# مرحله B، نسخه ۲ — تشخیص Routing با کنترل اختلاف عمقِ آرشیوی

در فایل `depth_drift_report.json` چهار plan زیر بررسی شدند:

- `system9.qasm`: P5 و P4 — عمق خام قبلی/فعلی ۱۷۳/۱۷۳؛ عمق نرمال‌شده قبلی/فعلی ۱۷۵/۱۷۷
- `sym9_146.qasm`: P8 و P4 — عمق خام قبلی/فعلی ۱۹۵/۱۹۵؛ عمق نرمال‌شده قبلی/فعلی ۲۰۰/۲۰۲

در هر چهار مورد تعداد گیت‌های اصلی، reset، measurement و conditional-X درست گزارش شده و Normalizer قدیمی و فعلی روی مدار فعلی عمق یکسانی می‌دهند. علت ریشه‌ای اختلاف ۲ لایه از این داده‌ها به‌تنهایی مشخص نیست؛ بنابراین اعداد Routed Depth اجرای جدید را **نباید با اعداد آرشیوی یکی دانست یا در تحلیل آماری مخلوط کرد**.

## فرق این نسخه با نسخه قبلی

کنترل عمق **حذف نشده**: پیش از کامپایل، فایل گزارش مستقلِ تشخیصی خوانده می‌شود و اثر انگشت SHA-256 فایل GidNET فعلی، QASMهای اصلی و CSVهای آزمایش، به‌همراه نسخه Qiskit تطبیق داده می‌شوند. همه ۴ plan باید در گزارش معتبر باشند و تنها اختلاف یکنواخت `+2` در عمق نرمال‌شده پذیرفته می‌شود. هنگام کامپایلِ دوباره، همان عمق خام و نرمال‌شدهٔ ثبت‌شده در گزارش نیز بازبینی می‌شود. اگر فایلی تغییر کرده باشد یا موردی اختلاف داشته باشد، برنامه متوقف می‌شود.

این بسته صرفاً برای همین تشخیص و همین چهار plan طراحی شده و یک گزینه عمومی «نادیده گرفتن اختلاف عمق» نیست.

## نصب

فایل ZIP را در **پوشه اصلی GidNET** استخراج کن تا این ساختار به دست بیاید:

```
GidNET-Qubit-Reuse-Algorithm/
  targeted_routing_diagnostics_v2_package/
    targeted_routing_diagnostics_v2.py
    README_FA.md
  GPT/
  reuse_validity.py
  gidnet/
  stage_b_depth_diagnosis/
    depth_drift_report.json
```

تست رگرسیون قبل از Routing (اگر همان فایل‌های تست مرحله قبل را هنوز داری):

```powershell
python -m pytest -q test_semantic_correctness.py test_reset_boundary_regression.py
```

## ۱. Dry-run بدون Routing

از پوشه اصلی GidNET و محیط `.venv_gidnet` فعال:

```powershell
python .\targeted_routing_diagnostics_v2_package\targeted_routing_diagnostics_v2.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --depth-report ".\stage_b_depth_diagnosis\depth_drift_report.json" `
  --out-dir ".\targeted_routing_results_v2" `
  --dry-run
```

باید پیام `DRY-RUN PASS` نشان داده شود. همچنین `compiled_plan_audit.json` و مدارهای نرمال‌شده در پوشه خروجی جدید ساخته می‌شوند.

## ۲. اجرای واقعی مرحله B

فقط اگر Dry-run و تست‌های Regression موفق بود، همان دستور را **بدون `--dry-run`** اجرا کن:

```powershell
python .\targeted_routing_diagnostics_v2_package\targeted_routing_diagnostics_v2.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --depth-report ".\stage_b_depth_diagnosis\depth_drift_report.json" `
  --out-dir ".\targeted_routing_results_v2"
```

۳ seed `100,101,102`، دو مدار، دو plan و سه حالت Layout => جمعاً ۳۶ اجرای Transpile.

- `free`: خود SABRE جای اولیه را انتخاب می‌کند.
- `anchor_better`: انتخاب اولیه از plan بهتر **طبق برچسب آرشیوی**، سپس نگاشت chainها.
- `anchor_worse`: برعکس، Anchor از plan بدتر آرشیوی.

برچسب‌های «بهتر» و «بدتر» براساس آزمایش‌های قدیمی‌اند؛ در نتیجه‌های جدید ممکن است رتبه عوض شود. Layout ثابت فقط **جای اولیه** را یکسان‌سازی می‌کند و راه‌حل داخلی SABRE را ثابت نمی‌کند.

## خروجی‌ها

از پوشه `targeted_routing_results_v2` این چهار فایل را بفرست:

- `REPORT.md`
- `mode_summary.csv`
- `paired_seed_differences.csv`
- `routing_runs.csv`

اگر خطا رخ داد، متن کامل خطا و `compiled_plan_audit.json` را هم بفرست؛ Source GidNET را تغییر نده. برای ادامهٔ اجرای نیمه‌تمام، با همان فایل‌ها و آرگومان‌ها `--resume` بزن. اگر نسخه کد یا ورودی‌ها تغییر کردند، برای جلوگیری از مخلوط شدن نتایج از پوشهٔ خروجی تازه استفاده کن.

## محدودیت‌ها

این نسخه روی داده‌های آرشیوی و گزارش تشخیصی تست‌های کنترل غیر Qiskit را پاس کرده، ولی Routing واقعی اینجا اجرا نشده (Qiskit روی محیط سازنده موجود نیست). یکسان بودن تعداد گیت‌ها، resetها و عمق‌ها **اثبات هم‌ارزی معنایی مدارها نیست**؛ پیش از نتیجه‌گیری پژوهشی، تست‌های End-to-end Semantic Correctness و Regression باید برقرار بمانند. `swap_critical_count` نیز تحلیل انتزاعیِ گراف DAG با مدت‌زمان واحد است، نه اطلاعات کالیبراسیون واقعی دستگاه.
