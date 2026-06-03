# ЛР №5 — CNN для FashionMNIST

Файлы:
- `Lab5_convnet.ipynb` — единый ноутбук (всё в одном: данные, обучение, метрики).
- `Lab5_convnet_train.ipynb` — обучение в Colab (GPU), сохраняет `best_convnet.pt` и `history.json`.
- `Lab5_convnet_infer.ipynb` — локальный инференс: метрики, confusion matrix, примеры.

## Что делаем

Классификация **FashionMNIST** (10 классов одежды, 28×28, ч/б). Цель — >95% accuracy.
Архитектура — компактная VGG-подобная свёрточная сеть с BatchNorm и Dropout.

## Данные

| | |
|---|---|
| train | 60 000 |
| test | 10 000 |
| классов | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| картинка | (1, 28, 28), нормировка `mean=0.2860, std=0.3530` |
| batch | 128 |
| аугментации (train) | `RandomHorizontalFlip(p=0.5)`, `RandomCrop(28, padding=4)` |

## Архитектура (3 свёрточных блока + FC-голова)

```
[Conv 1→32, 3×3, pad=1] → BN → ReLU
[Conv 32→32, 3×3, pad=1] → BN → ReLU
MaxPool 2×2 → Dropout(0.25)              # 28 → 14

[Conv 32→64, 3×3, pad=1] → BN → ReLU
[Conv 64→64, 3×3, pad=1] → BN → ReLU
MaxPool 2×2 → Dropout(0.25)              # 14 → 7

[Conv 64→128, 3×3, pad=1] → BN → ReLU
[Conv 128→128, 3×3, pad=1] → BN → ReLU
MaxPool 2×2 → Dropout(0.25)              # 7 → 3

Flatten (128·3·3 = 1152)
Linear(1152 → 256) → BN1d → ReLU → Dropout(0.5)
Linear(256 → 10)                          # логиты
```

- **6 свёрточных слоёв** (по 2 в каждом из 3 блоков) + **2 полносвязных**.
- Все Conv с `kernel=3, padding=1` → пространственный размер сохраняется внутри блока, уменьшается только в `MaxPool 2×2`.

## Гиперпараметры обучения

| | |
|---|---|
| эпох | 20 |
| optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| scheduler | `CosineAnnealingLR(T_max=20)` |
| loss | CrossEntropyLoss |
| чекпойнт | сохраняется по лучшей test acc → `best_convnet.pt` |

Итог: test accuracy ≈ **0.94–0.95** (FashionMNIST сложнее обычного MNIST из-за похожих классов Shirt/T-shirt/Pullover/Coat).

---

## Что делает каждая функция

**Слои сети**
- `nn.Conv2d(in_c, out_c, k, padding)` — свёртка; обучаемые веса = ядра + bias. `padding=1` при `k=3` сохраняет H×W.
- `nn.BatchNorm2d(c)` — нормирует активации по батчу для каждого канала; ускоряет обучение, чуть регуляризует.
- `nn.ReLU(inplace=True)` — нелинейность; `inplace` экономит память.
- `nn.MaxPool2d(2)` — берёт максимум в окне 2×2 → размер уменьшается вдвое.
- `nn.Dropout(p)` / `nn.Dropout2d` — зануляет нейроны/каналы с вероятностью p (только в `train`).
- `nn.Flatten()` — превращает `(B, C, H, W)` → `(B, C·H·W)`.
- `nn.Linear(in, out)` — полносвязный слой.
- `nn.BatchNorm1d(n)` — то же что 2d, но для одномерных активаций (после Flatten).

**Данные**
- `torchvision.datasets.FashionMNIST(root, train, download, transform)` — стандартный датасет.
- `transforms.Compose([...])` — последовательность преобразований картинки.
- `transforms.ToTensor()` — `PIL/numpy` → `torch.FloatTensor` и `[0, 255] → [0, 1]`.
- `transforms.Normalize(mean, std)` — `(x - mean) / std`.
- `transforms.RandomHorizontalFlip(p)` / `RandomCrop(28, padding=4)` — аугментации, искусственно расширяют train.
- `DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)` — итератор по батчам; `num_workers` — фоновые процессы для загрузки, `pin_memory` ускоряет копирование на GPU.

**Обучение**
- `loss_fn = nn.CrossEntropyLoss()` — softmax+NLL в одном; принимает **логиты** и индексы классов.
- `torch.optim.Adam(params, lr, weight_decay)` — адаптивный оптимизатор; `weight_decay` = L2.
- `CosineAnnealingLR(optimizer, T_max=N)` — плавно снижает `lr` по косинусу от стартового к ~0 за `N` эпох.
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — стандартный шаг.
- `scheduler.step()` — продвинуть scheduler на эпоху.
- `model.train()` / `model.eval()` — переключают режим Dropout/BatchNorm (на eval Dropout выключен, BN использует накопленные среднее/дисперсию).
- `with torch.no_grad():` — не строить граф (быстрее, нет градиентов; нужно для валидации/инференса).
- `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))` — сохранение/загрузка весов.

**Метрики и визуализация (infer)**
- `logits.argmax(dim=1)` — предсказанный класс.
- accuracy по классам — отдельная маска `y == cls_id` для каждого класса.
- **Confusion matrix** `cm[i, j]` — сколько объектов истинного класса `i` были предсказаны как `j`. Нормировка по строкам — доли от каждого истинного класса.
- `torch.softmax(logits, dim=1)` — вероятности классов; используются для поиска «уверенных ошибок» (high confidence, wrong prediction).

## Возможные вопросы

- **Почему 3 свёрточных блока, а не 2/4?**
  Картинка 28×28, после 3 MaxPool 2×2 пространство ужимается до 3×3 — дальше пулить нечего. С 2 блоками мало receptive field и параметров; 4 блока не влезут без `padding` ухищрений.
- **Зачем по 2 свёртки в блоке?**
  Два `3×3` подряд имеют такой же rec.field, как один `5×5`, но меньше параметров и больше нелинейностей. Стандартный приём из VGG.
- **Что даёт BatchNorm?**
  Нормирует распределение активаций, позволяет учить с бóльшим `lr`, ускоряет сходимость, действует как лёгкая регуляризация.
- **Зачем Dropout после каждого блока + 0.5 на FC?**
  Свёртки переобучаются медленнее, поэтому 0.25; FC — самая «капризная» часть с большим числом параметров, нужен сильный dropout 0.5.
- **Почему `padding=1` при `kernel=3`?**
  Чтобы свёртка не «съедала» рамку: `28 → 28`, и за уменьшение размерности отвечает только пулинг.
- **Зачем normalize на mean/std FashionMNIST?**
  Центрирует входы около нуля → градиенты адекватные, BatchNorm работает с нуля корректнее.
- **Зачем `CosineAnnealingLR`?**
  Большой `lr` в начале для быстрого спуска, малый — в конце для точной настройки. Без него к 20-й эпохе обычно «прыгает» вокруг минимума.
- **Зачем сохранять best по test accuracy, а не последний?**
  Из-за scheduler/Dropout последняя эпоха не обязательно лучшая.
- **Чем CrossEntropyLoss отличается от NLLLoss?**
  CE = `LogSoftmax + NLLLoss`. NLL ждёт уже log-вероятности.
- **Что значит `(C, H, W)` против `(H, W, C)`?**
  PyTorch — `channels first` (1, 28, 28). PIL/matplotlib — `channels last`. `permute(1, 2, 0)` при отрисовке.
- **Почему `num_workers=2`?**
  Параллельная подгрузка батчей в фоне; нужна на GPU, чтобы GPU не простаивал в ожидании данных.
- **Какая итоговая accuracy и где «теряется»?**
  ~0.94–0.95. По confusion matrix видно: путаются Shirt ↔ T-shirt/Pullover/Coat — это близкие классы.
