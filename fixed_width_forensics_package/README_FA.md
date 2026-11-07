# آزمایش مقایسهٔ ساختاری reuse planهای هم‌عرض — نسخهٔ اجرایی

## سؤال پژوهشی
برای **یک مدار اولیهٔ ثابت**، چرا reuse planهای هم‌عرض بعد از routing هزینه‌های متفاوتی دارند؟

این اسکریپت ابتدا بهترین و بدترین reuse plan را با میانگین ۳۰ اجرای ذخیره‌شدهٔ SABRE پیدا می‌کند؛ سپس ویژگی‌های **قابل محاسبه پیش از routing** را از QASM اصلی و reuse chainها می‌سازد. این کار یک تحلیل اکتشافی برای پیدا کردن ایدهٔ Repair Operator است، **نه یک الگوریتم Candidate Selection**.

## چه چیزی از ZIP ارسالی واقعاً قابل اجرا بود؟

در `GPT.zip`، CSVهای ۵۰ plan مربوط به پنج benchmark وجود دارند، ولی تنها فایل `system9.qasm` از بین QASMهای داخل ZIP با این benchmarkها **هم‌نام** است. به همین خاطر:

- مقایسهٔ بهترین/بدترین plan و شاخص‌های **موجود از Experiment 2** برای هر پنج مدار انجام شده؛
- استخراج ویژگی‌های **جدید از گیت‌ها و ساختار واقعی QASM** فقط برای ۱۳ plan مدار `system9` انجام شده؛
- برای چهار مدار دیگر هیچ ویژگیِ مبتنی بر QASM از روی فایل نامرتبط ساخته نشده است.

وقتی چهار QASM زیر را از پوشهٔ benchmark اصلی در کنار بقیهٔ ورودی‌ها بگذاری یا مسیر آنها را با `--qasm-dir` بدهی، همین کد بدون تغییر آنها را هم تحلیل می‌کند:

`alu-bdd_288.qasm`, `mini_alu_305.qasm`, `rd73_140.qasm`, `sym9_146.qasm`

## اجرا روی Windows / PowerShell

۱. ZIP اصلی خودت (`GPT.zip`) را Extract کن تا پوشه‌ای به نام `GPT` داشته باشی که CSVها و QASMها داخلش باشند.

۲. کنار پوشهٔ `GPT`، فایل `structural_forensics.py` را قرار بده.

۳. در محیط Python پروژه (مثل `.venv_gidnet`) وابستگی‌ها باید موجود باشند:

```powershell
python -m pip install numpy pandas scipy matplotlib
```

۴. دستور زیر را اجرا کن:

```powershell
python .\structural_forensics.py --data-dir ".\GPT" --out-dir ".\structural_forensics_results"
```

اگر QASMهای اصلی سایر benchmarkها را هم داری، از این دستور استفاده کن:

```powershell
python .\structural_forensics.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --out-dir ".\structural_forensics_results_all"
```

> اسکریپت هیچ routing تازه و هیچ اجرای دوبارهٔ GidNET انجام نمی‌دهد. Qiskit هم برای این تحلیل اکتشافی لازم نیست.

## فایل‌های خروجی

- `REPORT.md`: خلاصهٔ یافته‌ها و محدودیت‌ها.
- `coverage.json`: مشخص می‌کند کدام benchmarkها واقعاً QASM متناظر داشته‌اند.
- `all_five_existing_feature_comparison.csv`: بهترین و بدترین plan هر پنج مدار با شاخص‌های موجود.
- `per_plan_features.csv`: ویژگی‌های تازهٔ محاسبه‌شده برای planهای QASMدار؛ به‌همراه نتایج ۳۰ routing seed از قبل ذخیره‌شده.
- `best_worst_comparison.csv`: شناسه‌ها، فاصلهٔ عمق و SWAP، و فاصلهٔ اطمینان توصیفی برای اختلاف عمق روی seedهای مشترک.
- `feature_differences.csv`: تفاوت‌های ساختاری بهترین و بدترین plan.
- `within_circuit_exploratory_correlations.csv`: رابطهٔ توصیفی ویژگی‌های جدید با عمق، فقط برای فرضیه‌سازی.
- `equal_logical_depth_controlled_pairs.csv`: جفت planهایی که `normalized logical depth` یکسان دارند (برای مقایسهٔ کنترل‌شده).
- `per_chain_diagnostics.csv`: بار دوکیوبیتی، تعداد partner و میزان partner switching هر reuse chain.
- `best_worst_aligned_chains.csv`: تطبیق chainهای بهترین/بدترین plan با بیشترین هم‌پوشانی logical qubitها؛ چون **شناسهٔ chain در دو plan معنی ثابتی ندارد**.
- `best_worst_reuse_boundary_changes.csv`: reuse edgeهای مشترک و متفاوت بهترین/بدترین plan.
- `quotient_interaction_edges.csv`: تعداد gateهای دوکیوبیتی بین هر جفت reuse chain.
- `figures/`: نمودار رتبه‌بندی depth، heatmap ویژگی‌ها و scatterهای اکتشافی (هر کدام در یک شکل جدا).

## تعریف چند ویژگی جدید

- `partner_switch_rate`: در ترتیب گیت‌های دوکیوبیتی مربوط به هر chain، چند بار chainِ طرف مقابل عوض می‌شود؟ فرضیهٔ آزمایشی: جابه‌جایی پی‌درپی طرف تعامل شاید بر routing فشار وارد کند.
- `mean_pair_run_length`: طول متوسط دسته‌های پشت سرهم تعامل با یک partner ثابت.
- `critical_reuse_boundaries`: چند reuse edge به یکی از مسیرهای بحرانی DAGِ قبل از routing تعلق دارند؟
- `pre_routing_asap_gate_depth`: عمق DAG پس از افزودن reuse precedenceها، با فرض مدت یکسان برای تمام گیت‌ها؛ **با normalized logical depth دقیق Qiskit یکی نیست**.
- `mean_window_2q_chain_fraction`: در بازه‌های نسبی زمانیِ DAG بدون routing، چه سهمی از تعامل‌های هر chain در شلوغ‌ترین بازهٔ خودش قرار می‌گیرد؟

### کنترل‌های اجباری در کد

- identity تمام planها بین CSVهای Experiment 1، validation 30 seed و Experiment 2 باید یکسان باشد؛
- میانگین‌های فایل خام ۳۰ seed باید با خلاصهٔ planها تطابق داشته باشند؛
- width و مجموعهٔ logical qubitهای فعال با plan همخوان باشد؛
- reuse edgeها نباید در DAG گیت‌ها dependency cycle بسازند؛
- `max_weighted_degree`, `weighted_degree_cv` و `quotient_unique_edges` که مستقل محاسبه شده‌اند باید **دقیقاً** با Experiment 2 یکی باشند؛ در غیر این صورت اجرا با خطا متوقف می‌شود.

## نتیجهٔ فعلی و محدودیت تفسیر

این بسته روی `system9.qasm` اجرا شده؛ بهترین plan از نظر routed depth برابر `P11` و بدترین `P1` است. در جفت کنترل‌شدهٔ `P5` و `P4` هم `normalized logical depth`، `max weighted degree` و تقریباً `weighted degree CV` یکسان‌اند، ولی routed depth یکسان نیست.

به‌خصوص در تمام ۱۳ planِ `system9`، همبستگی رتبه‌ای `max_weighted_degree` با میانگین routed depth تقریباً صفر بوده است. بنابراین اگرچه بهترین/بدترین plan در پنج مدار hotspot متفاوتی دارند، این مشاهده به معنی توانایی پیش‌بینی مطمئنِ ترتیب تمام planها نیست.

**هیچ‌کدام از ویژگی‌های جدید در این مرحله به‌عنوان عامل علّی یا راهنمای قابل اتکای Repair تأیید نشده‌اند.** نمونه کم است، ویژگی‌ها اکتشافی هستند و انتخاب planهای extreme براساس همان نتایج routing انجام شده؛ برای ارزیابی فرضیه باید این تحلیل را روی چند QASM دیگر و داده‌های مستقل تکرار کنیم.

اسکریپت مخصوص QASM 2 با **یک qreg و گیت‌های استاتیک** است؛ روی reset، mid-circuit measurement، barrier و دستورهای کنترل شرطی عمداً خطا می‌دهد تا به‌جای تحلیل غلط بی‌صدا متوقف شود. این برنامه اعتبارسنج معنایی مدار نیست؛ Semantic Correctness را جداگانه بررسی کنید.
