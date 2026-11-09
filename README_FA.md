# آزمایش سوم: Nested Group Cross-Validation برای Repair/No Repair

**این مرحله بدون Routing جدید انجام می‌شود.** همهٔ مدل‌ها فقط با داده‌های `ml_train_prepare_v1` از ۲۸ مدار Training آموزش و ارزیابی می‌شوند. فایل تقسیم‌بندی قفل‌شده صرفاً برای کنترل هویت خانواده‌ها خوانده می‌شود؛ هیچ Label از Validation یا Final Test خوانده یا محاسبه نمی‌شود.

## محل استخراج ZIP

فایل‌های زیر را داخل `pilot_code` قرار دهید؛ فایل‌های قدیمی را پاک نکنید:

```text
pilot_code/
  nested_pairwise_repair_v3.py    # جدید
  test_nested_pairwise_v3.py      # جدید
  train_pairwise_repair_v2.py     # از مرحله قبل
  train_reuse_ranking_v1.py      # از مرحله قبل
  ml_train_prepare_v1/
    labels.csv
    routing_results.csv
    compiled_features.csv
    features.csv
    feature_pairs.csv
  ml_split_v1/
    split_ledger.csv
```

## اجرای کنترل سریع

```powershell
python .\test_nested_pairwise_v3.py
```

انتظار: `[PASS] nested family folds ...`.

## اجرای آزمایش اصلی با ۶۰ ویژگی

```powershell
python .\nested_pairwise_repair_v3.py `
  --train-dir .\ml_train_prepare_v1 `
  --split-ledger .\ml_split_v1\split_ledger.csv `
  --out-dir .\ml_nested_cv_v3 `
  --feature-mode full `
  --outer-folds 5 `
  --inner-folds 3 `
  --harm-penalty 1.0
```

`max-candidate-pairs=140` و `margin=0.2` مانند مرحلهٔ قبلی ثابت می‌مانند. همچنین همان دو مدل Pairwise Logistic و Pairwise HGB با Gateهای `0.50`, `0.65`, `0.75` امتحان می‌شوند. برای Depth، قاعدهٔ بدون مدل `logical_depth_rule` نیز کاندیدای انتخاب سیاست است؛ `no_repair` برای هر دو هدف در دسترس است.

**از قبل ثابت کرده‌ایم** که انتخاب سیاست در حلقهٔ درونی با این معیار انجام شود:

```text
Utility = mean_family(actual absolute gain) - 1.0 * mean_family(actual harm magnitude)
```

در این تعریف `gain = Original physical cost - Candidate physical cost` و `harm magnitude = max(-gain, 0)` است. در واقع زیان علاوه بر منفی‌کردن Gain، یک جریمهٔ دیگر هم می‌گیرد. این Utility از مقادیر *فقط Foldهای داخلی* محاسبه می‌شود و SWAP و Routed Depth هیچ‌وقت در یک تابع ترکیب نمی‌شوند. واحد هر Utility همان واحد هدف مربوطه است. اگر روش‌ها مساوی شوند، ترتیب ساده‌تر از پیش تعیین‌شده استفاده می‌شود.

## دقیقاً Nested CV چه می‌کند؟

1. **Outer GroupKFold (۵ بخش):** مجموعه خانواده‌های Test هر Fold را کنار می‌گذارد. خانوادهٔ یک مدار هیچ‌وقت هم‌زمان در Train و Test همان Fold نیست.
2. **Inner GroupKFold (۳ بخش):** فقط روی خانواده‌های Outer-Train، مدل و Gate را از گزینه‌های از پیش تعیین‌شده انتخاب می‌کند. هیچ نتیجهٔ Outer-Test برای انتخاب استفاده نمی‌شود.
3. **Outer Evaluation:** روش منتخب Inner CV روی تمام Outer-Train دوباره Fit می‌شود و روی Outer-Testِ ندیده اجرا می‌شود. No Repair و قاعدهٔ Logical Depth و Ridge/HGB ثابت نیز روی همان Originalها محاسبه می‌شوند؛ Oracle صرفاً سقف خوش‌بینانه است.
4. **Final TRAIN-only policy:** پس از پایان Nested CV، با یک Inner CV تازه روی کل Training یک روش برای هر هدف انتخاب و قفل می‌شود. اگر روش Pairwise باشد، مدلش روی تمام Training Fit و در `joblib` ذخیره می‌شود. **این مدل هنوز روی Validation مستقل بررسی نشده است.**

نکته: Nested OOF عملکرد **رویه انتخاب مدل** را ارزیابی می‌کند. روش انتخاب‌شده روی کل Training ممکن است با روش منتخب در Foldهای خارجی متفاوت باشد؛ نباید این دو را معادل دانست. تنها ۱۴ خانوادهٔ مستقل Training داریم، بنابراین برآوردها و فاصله‌های Bootstrap در سطح خانواده ناپایدارند.

## خروجی‌های مهم

- `nested_learning_summary.json`: نتایج و هشدارهای روش‌شناختی
- `nested_method_comparison.csv`: سود مطلق/درصدی، harm count، regret و مقایسه با Heuristic
- `outer_fold_policies.csv`: برای هر Fold مشخص می‌کند Inner CV چه مدل و Gate انتخاب کرده
- `inner_policy_scores.csv`: تمام Utilityهای انتخاب **داخل Fold**، برای بازبینی بدون نشت
- `nested_outer_decisions.csv`: تصمیم نهایی و سود واقعی در هر Original و Fold خارجی
- `nested_family_bootstrap_exploratory.csv`: فاصله‌های اکتشافی Bootstrap در سطح خانواده
- `final_train_only_policy_lock.json`: روش نهایی منتخب فقط از Training؛ برای ورود بعدی به Validation
- `*_frozen_nested_policy_TRAIN_ONLY.joblib`: مدل نهایی فقط وقتی خروجی انتخاب Pairwise باشد

## فایل‌هایی که برای تحلیل مرحله بعد ارسال کنید

1. `ml_nested_cv_v3\nested_learning_summary.json`
2. `ml_nested_cv_v3\nested_method_comparison.csv`
3. `ml_nested_cv_v3\outer_fold_policies.csv`
4. `ml_nested_cv_v3\final_train_only_policy_lock.json`

اگر فایل `nested_outer_decisions.csv` را هم ارسال کنید می‌توان ضررها را در سطح Original به تفکیک مدار و خانواده تحلیل کرد.

## محدودیت‌ها

- هیچ Routing تازه، Seed مستقل جدید، یا برچسبی از Validation/Final Test اینجا استفاده نمی‌شود.
- احتمال Pairwise Gate، احتمالِ کالیبره‌شدهٔ سودمندبودن Repair نیست.
- ۶۰ Feature مربوط به **قبل از Routing** هستند؛ هزینه فیزیکی Label نباید وارد Feature شود.
- هر Candidate در یک مجموعهٔ ثابت و ازپیش‌نمونه‌گیری‌شده رتبه می‌گیرد؛ **Beam Search هنوز با مدل هدایت نمی‌شود**.
- Regressionهای Ridge و HGB فقط Baseline ثابت‌اند؛ پارامترهای آن‌ها در این Nested CV تنظیم نمی‌شود.
- تعداد خانواده‌ها اندک است. `Final Test` همچنان رزرو شده است؛ نباید چندباره روی آن تنظیم انجام دهیم.
- برای اجرای دوباره، یک `--out-dir` جدید انتخاب کنید تا نتایج قدیمی بازنویسی یا باهم مخلوط نشوند.

## وضعیت تست

- تست کوچک واحد روی Foldهای خانوادگی، تقارن و عدم‌استفاده از Labelهای Outer-Test در انتخاب سیاست: PASS.
- مسیر کامل روی Labelهای واقعی + Featureهای کامپایل‌شده با `outer=2, inner=2`: PASS.
- مسیر کامل ۶۰‌ویژگی روی Labelهای واقعی + **۲۵ Feature ساختاری مصنوعی** با `outer=2, inner=2`: PASS؛ این اجرا فقط تست کد است، نتایج عددی آن ارزش تجربی ندارند.
- اجرای واقعی `outer=5, inner=3, full` روی ۲۵ Feature ساختاری اصلی باید روی سیستم شما انجام شود.
