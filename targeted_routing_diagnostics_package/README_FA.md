# مرحله B — Targeted Routing Diagnostics

این آزمایش مکمل **Controlled-Pair Structural Forensics (مرحله A)** است. هدفش پیدا کردن سرنخ‌های مربوط به **رفتار واقعی Routing** برای دو جفت reuse plan با width و شاخص‌های ساختاری مشابه است؛ این یک آزمایش **تشخیصی/اکتشافی** است، نه اثبات علت اختلاف Routed Depth یا ارزیابی نهایی الگوریتم Repair.

## طرح آزمایش

دو جفت ثابت از Experiment 1:

| مدار | plan کم‌عمق‌تر در ۳۰ seed قبلی | plan پرعمق‌تر | width |
|---|---|---|---:|
| `system9.qasm` | P5 | P4 | 7 |
| `sym9_146.qasm` | P8 | P4 | 5 |

برای هر جفت سه شرط اجرا می‌کنیم:

1. **free:** SABRE خودش Layout و Routing رو انتخاب می‌کنه؛ مطابق تنظیمات اصلی Experiment 1.
2. **anchor_better:** Layout انتخاب‌شده توسط SABRE برای plan کم‌عمق‌تر، در seed مرجع `100`، مبنا قرار می‌گیره. برای plan دیگر، reuse chainها با بیشترین همپوشانی logical qubitها تطبیق داده می‌شن و **همون محل‌های فیزیکی** به chainهای متناظر اختصاص می‌یابه. سپس SABRE فقط Routing رو انجام می‌ده.
3. **anchor_worse:** همین روند با Layout مرجع plan پرعمق‌تر انجام می‌شه، تا آزمایش فقط به نفع یکی از planها anchor نشه.

برای هر شرط، **سه seed مشترک 100، 101، 102** استفاده می‌کنیم؛ ۲ مدار × ۲ plan × ۳ شرط × ۳ seed = **۳۶ بار transpile**. برای اطمینان بیشتر بعداً می‌تونیم seedها رو بیشتر کنیم. **GidNET برای پیدا کردن reuse plan جدید اجرا نمی‌شه.** planهای فریز‌شده‌ی CSV مستقیماً در فرایند کامپایل فعلی GidNET تزریق می‌شن.

Topology = `CouplingMap.from_heavy_hex(7)`، Routing = SABRE، Layout آزاد = SABRE، `optimization_level=1`؛ مثل پروتکل Experiment 1. Resetها مطابق baseline به `measurement + conditional X` تبدیل می‌شن.

**نکته مهم:** منظور از Layout ثابت این نیست که logical qubitهای دو plan یکی‌به‌یکی فیزیکی یکسان‌اند؛ آنها reuse assignment متفاوتی دارند. ما chainهای متناظر رو با **حداکثر همپوشانی اعضا** مچ می‌کنیم. اگر تطبیق مبهم باشه، برنامه متوقف می‌شه و نتیجه‌ی ظاهراً کنترل‌شده نمی‌سازه. ثابت بودن Initial Layout هم به معنی یکسان شدن تمام تصمیم‌های بعدی SABRE نیست.

## نصب و اجرا روی ویندوز

ZIP رو در پوشه اصلی پروژه GidNET استخراج کن؛ دقیقاً کنار `GPT` و `.venv_gidnet`.

در PowerShell، پس از فعال‌سازی محیط:

```powershell
.\.venv_gidnet\Scripts\Activate.ps1

# اختیاری: تست‌های سبک خود اسکریپت
python -m unittest discover -s .\targeted_routing_diagnostics_package -p "test_targeted_routing_diagnostics.py" -v

# مرحله اول: فقط بازسازی و بررسی چهار plan؛ هیچ routing اجرا نمی‌شود
python .\targeted_routing_diagnostics_package\targeted_routing_diagnostics.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --out-dir ".\targeted_routing_results" `
  --dry-run
```

اگر خروجی `DRY-RUN PASS` بود، این بار بدون `--dry-run` اجرا کن:

```powershell
python .\targeted_routing_diagnostics_package\targeted_routing_diagnostics.py `
  --data-dir ".\GPT" `
  --qasm-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --out-dir ".\targeted_routing_results"
```

هر route بلافاصله در `routing_runs.csv` ثبت می‌شه. اگر وسط آزمایش قطع شد، همون دستور رو با **`--resume`** در آخرش تکرار کن. با تغییر نسخه Python/GidNET/Qiskit، seedها، داده‌ها یا خود کد، resume عمداً مجاز نیست؛ باید پوشه خروجی تازه انتخاب کنی.

**قبل از شروع:** چون قبلاً bug مربوط به reset گیت‌های دوکیوبیتی رو اصلاح کردی، مطمئن شو هنوز نسخه پچ‌شده‌ی `gidnet/qubitreuse.py` رو داری و تست‌های `test_semantic_correctness.py` و `test_reset_boundary_regression.py` پاس می‌شن. اسکریپت فایل GidNET رو **دستکاری نمی‌کنه**.

## خروجی‌ها

| فایل | محتوایش |
|---|---|
| `REPORT.md` | جدول اختلاف عمق/SWAP به تفکیک شرط Layout + محدودیت تفسیر |
| `mode_summary.csv` | میانگین delta (بدتر − بهتر) در seedهای مشترک برای هر شرط |
| `paired_seed_differences.csv` | تفاوت seed به seed در Depth، SWAP، SWAPهای مسیر بحرانی و لایه‌ها |
| `routing_runs.csv` | مشخصات هر routed circuit و Layout اولیه‌ی استخراج‌شده |
| `compiled_plan_audit.json` | تایید width، بازسازی ویژگی‌های Experiment 2، اثر انگشت plan و تعداد resetها |
| `routed_qpy/` | مدارهای Routed واقعی با فرمت QPY، یکی برای هر run |
| `diagnostics/*__events.csv` | هر عملیات فیزیکی، ASAP Layer و Critical-Path membership |
| `diagnostics/*__layers.csv` | تعداد SWAP/CX/Measurement و سایر عملیات در هر لایه |
| `diagnostics/*__edges.csv` | توزیع SWAP و CX بر روی یال‌های فیزیکی |
| `figures/` | تفاوت Routed Depth در seedهای مشترک، بین شروط مختلف |
| `config.json` | نسخه Qiskit و SHA256 کد GidNET، QASMها و CSVها برای reproducibility |

برای تحلیل اولیه **این چهار فایل** رو بفرست: `REPORT.md`، `mode_summary.csv`، `paired_seed_differences.csv` و `routing_runs.csv`. اگر نیاز بود بعداً QPY و event/layer CSV یک یا دو run مشخص رو می‌گیریم. نیازی نیست از اول همه فایل‌های حجیم QPY رو بفرستی.

## چک‌های اجباری پیش از Routing

- اصالت plan با تطبیق شناسه‌ها و ۳۰ seed ثبت‌شده‌ی قدیمی؛
- ساخت یک ReuseValidityChecker مستقل و قبولی هر چهار reuse plan؛
- تطبیق quotient metrics ساخته‌شده از QASM با `max_weighted_degree`، `weighted_degree_cv` و `quotient_unique_edges` آزمایش ۲؛
- کامپایل **همان** reuse plan و بررسی تعداد resetهای لازم؛
- تطبیق normalized logical depth با مقدار CSV قدیمی؛
- ثبت hash فایل GidNET نصب‌شده و QASM اصلی؛
- بررسی اینکه SABRE واقعاً Initial Layout در حالت‌های anchor رو بدون تغییر پذیرفته باشه.

اگر یکی از این‌ها برقرار نباشه، اجرای آزمایش متوقف می‌شه؛ **به زور تا مرحله Routing جلو نمی‌ره**.

## تفسیر علمی

اگر اختلاف Depth تحت هر دو anchor باقی بمونه، نمی‌تونیم اختلاف رو فقط به انتخاب Layout اولیه SABRE در همین anchorها نسبت بدیم. اما هنوز ثابت نکردیم کدوم interaction خاص علتشه؛ Routing و انتخاب مسیرهای SWAP همچنان stochastic هستن.

تعداد SWAPهای **روی مسیر بحرانی DAG کامپایل‌شده** و تعداد **لایه‌های حاوی SWAP** می‌تونن بهتر از صرف SWAP Count نشون بدن چرا دو مدار عمق متفاوتی دارن. این معیارها پس از Routing استخراج می‌شن و **پیش‌بینی‌گر قبل از Routing نیستن**. اگر سرنخی پیدا کردیم باید بعداً یک proxy پیش از Routing برای آن طراحی و آزمایش کنیم.

**توجه:** ابزار Qiskit در محیط ساخت این بسته نصب نبود؛ بنابراین خود Routing و Dry-run چهارتایی رو اینجا اجرا نکردم. کد Python از نظر syntax و شش تست بدون نیاز به Qiskit بررسی شده، ولی اعتبار اجرای Qiskit فقط پس از نتیجه‌ی Dry-run روی سیستم تو مشخص می‌شه. در صورت خطا خروجی رو بفرست تا روی همون مشکل مشخص اصلاح کنیم.
