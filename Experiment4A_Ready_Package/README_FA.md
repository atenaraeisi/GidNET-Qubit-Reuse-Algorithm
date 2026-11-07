# Experiment 4A — Physical Value of Expanded One-Step Neighborhood

هدف: بررسی اینکه گسترش همسایگی یک‌مرحله‌ای reuse plan با `chain_transfer` و سایر operatorها، **فرصت رسیدن به SWAP / Routed Depth بهتر** را نسبت به `suffix_swap` ایجاد می‌کند یا نه.

**مهم:** این آزمایش یک «opportunity study» است. مجموعهٔ همسایه‌های بزرگ‌تر از نظر بهترین نتیجه همیشه حداقل به خوبی مجموعهٔ کوچک‌تر است؛ پس باید **تعداد موارد بهبود واقعی، اندازهٔ بهبود، و پایداری روی seedهای تازه** گزارش شوند. این آزمایش هنوز Budgeted Repair یا اثبات برتری بر Restart نیست.

## فایل‌های داخل بسته

- `experiment4A_physical_value.py`: کد اصلی (prepare → explore → validate).
- `Experiment4A_Run.ipynb`: رابط اجرای مرحله‌ای داخل Jupyter.
- `data/adversarial_mutation_cases.csv`: تمام ۸۶۴۲ تغییر Full Coverage؛ ۴۸۵ مورد دارای اعتبار ایستای مثبت.
- `data/width_and_original_plan_audit.csv`: اطلاعات ۵۰ Plan اولیه و Width آن‌ها.
- `data/audit_report.json`: گزارش اعتبارسنجی که تأیید می‌کند ۸۶۴۲ مورد بررسی شده‌اند.

در نسخه فعلی، `prepare` اجرا و آزموده شده و انتظار داریم مجموع همسایه‌های معتبر یکتای به‌ازای هر Plan برابر ۲۹۱ باشد (۱۳۰ suffix، ۱۵۹ مورد جدید از chain transfer، ۲ مورد دیگر).

## فایل‌های ضروری برای اجرای Routing

در پوشهٔ پروژهٔ اصلی لازم است این موارد موجود باشند:

1. `experiment2_results/experiment2_plan_features.csv` با ستون‌های `circuit, plan_id, width, reuse_chains` (برای به‌دست‌آوردن **اصلِ** ۵۰ Plan؛ فایل Full Coverage خودِ Planهای اصلی را ذخیره نمی‌کند).
2. پوشهٔ QASM حاوی این پنج فایل: `alu-bdd_288.qasm`، `mini_alu_305.qasm`، `rd73_140.qasm`، `sym9_146.qasm` و `system9.qasm`.
3. نسخهٔ اصلاح‌شدهٔ پروژهٔ GidNET شامل `gidnet/qubitreuse.py` و `gidnet/reset_lowering.py` همراه وابستگی‌ها. یا یک فایل Python با توابع `build_dynamic_circuit(qc, chains)` و در صورت نیاز `normalize_for_routing(qc)` برای استفاده از **همان pipeline آزمایش ۳**.

از Python/محیط آزمایش ۳ (Qiskit 0.45.x) استفاده کن. هنگام مقایسه مستقیم با آزمایش ۳ مهم است که تبدیل Plan به مدار، Measurement/Reset، تنظیمات transpilation، Qiskit و GidNET با نسخهٔ آزمایش ۳ یکسان باشند.

## اجرا با PowerShell در ریشهٔ پروژه GidNET

پوشهٔ محتوای ZIP را در ریشهٔ پروژه بریز یا مسیر فایل‌ها را کامل وارد کن. دستورات نمونه در صورتی درست هستند که فایل‌ها در همان پوشهٔ فعلی باشند:

```powershell
python experiment4A_physical_value.py prepare --mutations "data/adversarial_mutation_cases.csv" --audit "data/width_and_original_plan_audit.csv" --out "experiment4a_results"
```

برای شروع Routing، مسیر QASM را با مسیر واقعی کامپیوترت جایگزین کن:

```powershell
python experiment4A_physical_value.py explore --mutations "data/adversarial_mutation_cases.csv" --audit "data/width_and_original_plan_audit.csv" --plans "experiment2_results/experiment2_plan_features.csv" --bench-dir "C:\path\to\CaQR-main\benchmarks" --out "experiment4a_results"
```

وقتی Explore کامل شد:

```powershell
python experiment4A_physical_value.py validate --mutations "data/adversarial_mutation_cases.csv" --audit "data/width_and_original_plan_audit.csv" --plans "experiment2_results/experiment2_plan_features.csv" --bench-dir "C:\path\to\CaQR-main\benchmarks" --out "experiment4a_results"
```

**مرحله Explore حدود `تعداد Planهای متمایز × ۵ seed` بار transpile انجام می‌دهد.** اسکریپت نتایج هر اجرا را در `routes_train.csv` ذخیره می‌کند و قابل ادامه دادن (resume) است. برای تست کوتاه می‌توانی `--max-new-routes 5` را اضافه کنی، سپس بدون آن اجرا کنی.

مرحله Validate فقط انتخاب‌های Freeze شده از Explore (و Originalها) را با Seedهای **۱۰۰ تا ۱۲۹** دوباره Route می‌کند؛ نتایج `holdout` برای انتخاب مجدد Plan استفاده نمی‌شوند.

## پروتکل و گزارش‌ها

1. `suffix_only`، `suffix_plus_chain` و `all_operators` سه مجموعهٔ تو در تو هستند.
2. نتایج یک Plan متمایز (بعد از canonicalization ترتیب کل chainها) به‌ازای هر circuit و seed فقط یک بار محاسبه می‌شوند و بین Planهای اولیه reuse می‌شوند.
3. برای هر یک از ۵۰ Original، بهترین همسایه برای **SWAP** و بهترین همسایه برای **Depth** **مستقل از یکدیگر** انتخاب می‌شود. با انتخاب بهتر از Original (یا نگه‌داشتن Original) گزارش «فرصت بهبود» به‌دست می‌آید.
4. حالت بدون محدودیت depth و با شرط حداکثر **۵٪ افزایش Logical Depth** هر دو گزارش می‌شوند.
5. روی Seedهای تازه، همان انتخاب فریز‌شده ارزیابی می‌شود؛ مقایسهٔ زوجی با Original و Suffix-only گزارش می‌شود.

خروجی‌های کلیدی:

- `coverage_per_source.csv`: تعداد همسایه‌ها برای هر Original
- `candidate_catalog.csv` و `source_memberships.csv`: کاندیداها و اینکه با کدام operator به دست آمده‌اند
- `candidate_routing_means.csv`: میانگین physical metrics روی Seedهای Explore
- `frozen_winners.csv`: Planهای انتخاب‌شده پیش از Seedهای تازه
- `opportunity_summary.csv` و `source_opportunity_comparison.csv`: تعداد و مقدار بهبود نسبت به suffix
- `fresh_seed_pairs.csv` و `fresh_seed_comparison.csv`: اعتبارسنجی روی Seedهای جدید
- `routing_settings.json`: مشخصات config و fingerprint ورودی‌ها

## محدودیت‌های مهم

- Full Coverage فقط **اعتبار وابستگی ایستا** را چک کرده؛ درست بودن Semantics نهایی Measurement/Reset با آن ثابت نشده. پیش از ادعای اجرای فیزیکی درست، بررسی مستقل Semantics/Reset-lowering ضروری است.
- اجرای `explore` با روش پیش‌فرض `GidNET` در این فایل بر مبنای override کردن تولید reuse sequence و فرستادن Plan ذخیره‌شده به `compile_to_dynamic_circuit` است. برای **مقایسهٔ دقیق با Experiment 3** باید با نوت‌بوک قبلی تطبیق و در صورت لزوم با `--adapter` جایگزین شود.
- متریک `swap_count` در اینجا تعداد `swap` پس از transpile با تنظیمات تعیین‌شده است، و `routed_depth` عمق مدار پس از transpile. اگر آزمایش ۳ از تعریف متفاوتی استفاده کرده، باید یکسان شوند.
- نتیجه هنوز **بودجه یکسان** با Restart یا Suffix Repair را مقایسه نمی‌کند؛ این آزمایش فقط می‌گوید Plan بهتر در همسایگی گسترده وجود دارد یا خیر.
- مرحلهٔ Explore/Validate در این محیط اجرا نشده چون QASM، Planهای Original و Qiskit هم‌نسخهٔ پروژه موجود نبودند. نتایج فیزیکی تا اجرای واقعی روی سیستم تولید نمی‌شوند.
