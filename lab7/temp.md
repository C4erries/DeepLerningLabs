# ЛР №7 — Трансформеры (DistilBERT + IMDB)

Файлы:
- `Lab7_transformer.ipynb` — короткий отчётный ноутбук (кривые обучения).
- `Lab7_transformer_train.ipynb` — fine-tuning в Colab (T4 GPU). Сохраняет `distilbert_imdb_ft/` и `history.json`.
- `Lab7_transformer_infer.ipynb` — локальный инференс: метрики, PR-кривая, ROC-AUC, свои примеры.

## Что делаем

Бинарная классификация отзывов **IMDB** (negative / positive) через
**fine-tuning DistilBERT** — дистиллированной версии BERT (Hugging Face
`distilbert-base-uncased`). Используем готовый `Trainer` из `transformers`.

## Данные

| | |
|---|---|
| источник | `stanfordnlp/imdb` (HuggingFace Datasets) |
| train / test (полные) | 25 000 / 25 000 |
| **подвыборка для работы** | **4000 / 2000** (`seed=42`) — чтобы fine-tune укладывался в ~2–3 мин на бесплатной T4 |
| классов | 2 (`neg` / `pos`) |
| баланс | 50/50 |
| `max_length` токенизатора | 256 |

## Архитектура

```
Input text
    │
WordPiece tokenizer (max_length=256)
    │
DistilBERT backbone:
   • 6 encoder-слоёв трансформера (BERT — 12)
   • hidden = 768
   • 12 attention-голов на слой
   • ~66M параметров (BERT — ~110M)
    │   выход [CLS]-токена (768)
Pre-classifier: Linear(768 → 768) + ReLU + Dropout (это стандартная голова DistilBERT, есть из коробки)
    │
Classifier: Linear(768 → 2)
    │
Softmax → P(neg), P(pos)
```

Backbone предобучен Hugging Face'ом на masked language modeling (MLM, BookCorpus + Wikipedia)
через дистилляцию из BERT (~97% качества BERT на GLUE, в ~2 раза быстрее).

## Гиперпараметры fine-tuning

| | |
|---|---|
| эпох | 2 |
| batch size | 16 (train) / 32 (eval) |
| optimizer | AdamW (по умолчанию в `Trainer`) |
| learning rate | `2e-5` |
| weight_decay | `0.01` |
| scheduler | linear warmup → linear decay (по умолчанию у `Trainer`) |
| precision | fp16 (на GPU) |
| loss | CrossEntropyLoss (внутри модели) |

Итог: accuracy ≈ **0.88–0.90**, ROC-AUC ≈ **0.94–0.96**.

## Что делает каждая функция

**Данные и токенизация**
- `load_dataset("stanfordnlp/imdb")` — скачивает датасет с HuggingFace Hub в кэш.
- `.shuffle(seed=42).select(range(N))` — перемешивает с фиксированным seed и
  берёт первые N — детерминированная подвыборка.
- `AutoTokenizer.from_pretrained("distilbert-base-uncased")` — токенизатор того
  же типа, что и при обучении модели (WordPiece).
- `tokenizer(text, truncation=True, max_length=256)` — режет длинные тексты до
  256 токенов (хвосты длинных рецензий теряются).
- `dataset.map(tokenize, batched=True)` — применяет функцию ко всему датасету
  батчами. Кэшируется на диск.
- `DataCollatorWithPadding(tokenizer)` — паддит последовательности до длины
  самой длинной в батче (динамический паддинг, экономит вычисления).

**Модель**
- `AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, id2label, label2id)`
  — грузит backbone и докручивает голову `Linear → ReLU → Dropout → Linear(→num_labels)`.
  Голова инициализируется случайно (отсюда warning при загрузке).
- `model.to(device)` — перенос на GPU.

**Обучение через `Trainer`**
- `TrainingArguments(...)` — все гиперпараметры в одном объекте:
  - `num_train_epochs` — число эпох;
  - `per_device_train_batch_size` / `per_device_eval_batch_size` — батчи;
  - `learning_rate`, `weight_decay` — стандартные параметры AdamW;
  - `eval_strategy="epoch"` — eval на test после каждой эпохи;
  - `logging_steps=50` — записывать train loss каждые 50 шагов;
  - `save_strategy="no"` — не сохранять промежуточные чекпойнты (экономия диска);
  - `fp16=True` — mixed precision на GPU (быстрее и меньше памяти);
  - `report_to="none"` — не отправлять логи в wandb/etc.
- `compute_metrics(eval_pred)` — функция, которую `Trainer` вызывает при eval.
  Получает `(logits, labels)`, должна вернуть dict с метриками.
- `Trainer(model, args, train_dataset, eval_dataset, data_collator, compute_metrics)`
  — оборачивает всё; берёт на себя цикл обучения, eval, mixed-precision, scheduler.
- `trainer.train()` — запуск обучения.
- `trainer.state.log_history` — лог по шагам (train loss / eval metrics) → пишем в `history.json`.
- `trainer.save_model(dir)` — сохраняет веса + конфиг модели.
- `tokenizer.save_pretrained(dir)` — сохраняет токенизатор (обязательно рядом
  с моделью, иначе при загрузке инференса не на чем токенизировать).

**Инференс**
- `AutoModelForSequenceClassification.from_pretrained(dir)` — грузит обратно
  из локальной папки.
- `model.eval()` — выключает Dropout.
- `torch.softmax(logits, dim=-1)` — вероятности классов.
- `argmax(-1)` — предсказанный класс.

**Метрики**
- `accuracy_score`, `classification_report` — стандартные sklearn.
- `precision_recall_curve(y_true, y_score)` — точки (precision, recall) для всех
  порогов; `average_precision_score` — площадь под PR-кривой.
- `roc_curve(y_true, y_score)` — точки (FPR, TPR); `auc` — площадь под ROC-кривой.

## Возможные вопросы

- **Что такое трансформер?**
  Архитектура из «Attention Is All You Need» (2017), полностью построенная на
  self-attention, без рекуррентных и свёрточных слоёв.
- **Что такое self-attention?**
  Каждый токен «смотрит» на все остальные токены последовательности и взвешивает
  их вклад: `Attention(Q, K, V) = softmax(QKᵀ/√d) · V`. Q/K/V — линейные проекции
  входа.
- **Что такое multi-head attention?**
  Несколько параллельных голов attention с разными проекциями Q/K/V — учат
  разные виды зависимостей. У DistilBERT — **12 голов** на слой.
- **Зачем positional encoding?**
  Сам attention перестановочно-инвариантен (не знает порядок токенов).
  Позиционное кодирование добавляется к эмбеддингам, чтобы дать сети информацию
  о позициях.
- **Чем BERT отличается от GPT?**
  BERT — encoder-only, двунаправленный (видит и левый, и правый контекст),
  обучается на masked language modeling. GPT — decoder-only, авторегрессионный
  (видит только левый контекст), обучается на next-token prediction.
- **Что такое DistilBERT?**
  Дистиллированная версия BERT: 6 encoder-слоёв вместо 12, ~66M параметров против
  ~110M, ~97% качества на GLUE, в ~2× быстрее. Обучен student-моделью с loss'ом
  «копировать выходы BERT».
- **Сколько слоёв и параметров в нашей модели?**
  6 трансформер-слоёв, hidden=768, 12 голов, ~66M параметров. Плюс наша голова
  `Linear(768→768) + Linear(768→2)`.
- **Что такое `[CLS]`-токен?**
  Специальный токен, добавляемый в начало последовательности. Его выходной
  эмбеддинг используется как агрегированное представление всего текста для
  классификации.
- **Почему `max_length=256`?**
  Компромисс: средняя рецензия IMDB длиннее, но 512 удваивает время/память
  (attention квадратичная по длине). 256 хватает, теряем хвосты длинных
  рецензий.
- **Почему именно `lr=2e-5`?**
  Стандарт для fine-tuning BERT-подобных моделей (5e-5…1e-5). Слишком большой →
  ломает предобученные веса; слишком малый → не доучится за 2 эпохи.
- **Что такое AdamW?**
  Adam с правильно реализованным weight decay (в обычном Adam он смешивается
  с моментами и работает иначе). Стандарт для трансформеров.
- **Зачем fp16 (mixed precision)?**
  Половинная точность для большей части вычислений → быстрее и в 2× меньше
  памяти, без заметной потери качества.
- **Зачем разделение train/infer?**
  Обучение требует GPU и долгое; инференс — секунды на CPU. Сохраняем веса
  и грузим где удобно.
- **Что такое ROC-AUC и зачем он?**
  Площадь под ROC-кривой (TPR vs FPR при разных порогах). Не зависит от
  выбранного порога, оценивает качество разделимости классов.
- **Чем precision/recall лучше accuracy?**
  Accuracy на дисбалансе вводит в заблуждение. Precision = `TP/(TP+FP)`,
  Recall = `TP/(TP+FN)`. Здесь данные сбалансированы 50/50, поэтому accuracy
  и F1 близки.
- **Почему `id2label`/`label2id` важны?**
  Записывают семантические имена меток в конфиг модели; после загрузки модель
  правильно мапит индекс → строку без дополнительных «магических» констант.
