پایلوت ساخت دیتاست Multi-step Fixed-width Repair
=========================================

هدف:
در یک اجرای کنترل‌شده، از Original Planهای معتبرِ دو موج غربالگری، Candidateهای ۱ تا ۳ حرکت معتبر بسازیم، ۲۵ ویژگی پیش از Routing را ثبت کنیم و (فقط در صورت اجرای دستور route) نتایج دو Seed ابتدایی را برچسب بزنیم.

این نسخه برای تستِ پایپ‌لاین است؛ برای آموزش نهایی، مقایسه مدل‌ها، Semantic Equivalence عمومی، یا ارزیابی Held-out مناسب نیست.

پیش‌نیاز:
- محیط مجازی قبلی با Qiskit 0.45.3، numpy، pandas، networkx، retworkx.
- هر دو پوشه fixed_full و wave2 از غربالگری قبلی، داخل reuse_audit_fixed موجود باشند.
- فایل‌های QASM در مسیرهای اصلی خود در Windows باقی مانده باشند (به qasm_path داخل circuit_audit.csv رجوع می‌شود).
- فایل ZIP را داخل C:\Users\ASUS\Downloads\reuse_audit_fixed استخراج کنید.
- برنامه GidNET از نسخه qubitreuse.py ارسالی شما استفاده می‌کند (patchهای candidate-matrix و reset، بدون تغییر فایل اصلی). مسیر router_template.py همان Heavy-hex distance=7، SABRE، optimization_level=1 است.

۱) فعال کردن محیط مجازی قبلی (مطابق سیستم خود) و رفتن به پوشه:
   cd "C:\Users\ASUS\Downloads\reuse_audit_fixed"
   python -c "import qiskit; print(qiskit.__version__)"

۲) فقط تست تولید داده، برای یک مدار و یک Original (بدون Routing):
   python .\build_multistep_pilot.py prepare --audit-dirs .\fixed_full .\wave2 --out-dir .\pilot_smoke --circuits CaQR::cnt3-5_179.qasm --max-originals 1 --max-candidates 4

۳) پس از موفقیت، آماده‌سازی ۶ مدار (بدون Routing):
   python .\build_multistep_pilot.py prepare --audit-dirs .\fixed_full .\wave2 --out-dir .\multistep_pilot_out

۴) فقط اگر آماده‌سازی مرحله ۳ درست بود، Routing بسیار کوچک را صریحاً اجرا کنید:
   python .\build_multistep_pilot.py route --out-dir .\multistep_pilot_out --seeds 42,43 --max-route-groups 1 --max-route-candidates-per-group 1

۵) برچسب‌های نمونه را از Seedهای دقیق مرحله ۴ بسازید:
   python .\build_multistep_pilot.py labels --out-dir .\multistep_pilot_out --seeds 42,43

در صورت تأیید مراحل، route کاملِ همه Candidateهای این پایلوت فقط با دستور صریح زیر ممکن است:
   python .\build_multistep_pilot.py route --out-dir .\multistep_pilot_out --seeds 42,43

نکته‌های مهم:
- دستور prepare به طور پیش‌فرض فقط ۶ مدار منتخب را می‌خواند: ۳ نمونه CaQR و ۳ نمونه QASMBench؛ دو multiply_n13 از دو منبع عمداً جدا نگه داشته می‌شوند.
- هر مدار حداکثر ۲ Original و هر Original حداکثر ۲۰ Candidate برچسب‌پذیر می‌گیرد. پس برای دو Seed سقف نظری حدود ۵۰۴ اجرای Routing است؛ تکرار Planهای یکسان از Cache مؤثر استفاده می‌کند و تعداد واقعی ممکن است کمتر باشد. این عدد از خروجی manifest.json دقیق‌تر مشخص می‌شود.
- تعداد مسیرهای معتبر و فاصله هر Candidate تحت جست‌وجوی بودجه‌دار اندازه‌گیری می‌شود؛ discovery_move_depth الزاماً حداقل طول مسیر در فضای کامل نیست.
- برچسب درصدی وقتی baseline SWAP=0 باشد نامعین می‌ماند؛ اختلاف مطلق همچنان ثبت می‌شود.
- جست‌وجو برای انتخاب Candidate هیچ Routing انجام نمی‌دهد. Router فقط پس از انتخاب و در فرمان route اجرا می‌شود.
- فایل‌های Exp 4C-1، Exp 5 و آداپتر فعلی بدون تغییر نگه داشته شده‌اند؛ هدف پایلوت بازتولید دقیقی از کل فضای Exp5 نیست، بلکه نمونه‌گیری چندقدمی با بودجه محدود است.
- طرح Seedهای قبلی 42-46 و 100-129 برای ارزیابی فیزیکی سابق بوده است. در اینجا 42,43 فقط برای Smoke Test است؛ هیچ‌وقت از نتایج این پایلوت به عنوان عملکرد نهایی یا Training/Test تمیز استفاده نکنید.
- ممکن است برخی Candidateها با Compiler فیزیکی خطا بدهند؛ در routing_results.csv با status ثبت می‌شوند. بررسی معنایی مستقل برای مدارهای جدید باید جدا انجام شود.
- قبل از انتخاب مدل و تنظیم Hyperparameterها، Train/Validation/Test را در سطح مدار/خانواده تعریف و Splitهای نهایی را فریز کنید.

فایل‌های خروجی prepare:
- selected_candidates.csv  : هر Original به همراه Candidateهای منتخب (یکی از برنامه‌های آماده‌ی routing)
- discovery.csv            : همه Planهای مشاهده‌شده و والد/Operator/تعداد حرکت مسیر کشف
- plan_catalog.csv         : Planهای منتخب یکتا برای Cache Routing
- features.csv             : ۲۵ ویژگی ساختاری هر Plan منتخب
- feature_pairs.csv        : اختلاف ویژگی Original و Candidate، بدون نرمال‌سازی آگاه به Test
- source_summary.json      : آمار مسیریابی ساختاری هر Original
- prepare_errors.json      : خطاهای مدارهای ناموفق (باید بررسی شوند)
- manifest.json            : پروتکل، مسیر QASM، تعداد دقیق برنامه‌های Routing و نسخه‌ها

فایل‌های خروجی route و labels:
- routing_setup.json       : پیکربندی ثابت Router و Seedها
- routing_results.csv      : SWAP، Routed Depth، زمان و status برای هر Plan/Seed (قابل ادامه)
- compiled_features.csv    : logical_depth، norm_depth و dynamic width از همان compiler
- labels.csv               : gain مطلق و درصدی مستقل برای SWAP و Depth، به همراه No-repair original
- unlabeled.csv            : جفت‌هایی که همه Seedها یا status موفق ندارند
- label_summary.json       : تعداد Labelهای کامل و محدودیت آزمایش

اگر prepare با Error مواجه شد، manifest.json و prepare_errors.json را بفرستید.
اگر Routing با Error مواجه شد، routing_results.csv و compiled_features.csv را بفرستید.
