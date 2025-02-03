# AnimeFaceStyle

**AnimeFaceStyle** — это Telegram-бот [@AnimeStyleFace_bot](https://t.me/AnimeStyleFace_bot), который преобразует фотографии лиц в аниме-стиль с использованием нейросетевой модели CycleGAN.  
Проект разработан в рамках курса **Deep Learning School**.

## О проекте

Бот использует модель **CycleGAN** для переноса стиля изображений.  
CycleGAN — это разновидность генеративно-состязательных сетей (GAN), которая способна преобразовывать изображения между двумя различными доменами без необходимости в парных данных.  
В данном случае происходит преобразование человеческих лиц в аниме-стиль.

## Датасеты

Для обучения модели использовались следующие датасеты:

- [Anime Faces Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) — содержит изображения аниме-лиц, используемых для стилизации.
- [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) — включает фотографии реальных лиц, использованных для обучения модели преобразования.

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/serttyzar/AnimeFaceStyle.git
cd AnimeFaceStyle
```

### 2. Установка зависимостей

Убедитесь, что у вас установлен **Python 3.7+**, и выполните установку зависимостей:

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

Создайте файл `.env` в корневой директории проекта и добавьте в него ваш токен Telegram-бота:

```ini
TELEGRAM_TOKEN=ваш_токен
```

### 4. Запуск бота

После установки зависимостей запустите Telegram-бота:

```bash
python bot.py
```

## Архитектура проекта

### Основные компоненты

- **Модель CycleGAN**  
  Используется для преобразования изображений между доменами (реальные лица → аниме-стиль).

- **Telegram-бот**  
  Принимает изображения от пользователей, обрабатывает их и отправляет обратно в виде стилизованных картинок.

- **Обработчик изображений**  
  Загружает входные изображения, подготавливает их для обработки моделью и сохраняет результат.

### Основные файлы

- `bot.py` — основной файл с логикой Telegram-бота.
- `model.py` — код, загружающий и применяющий предобученную модель CycleGAN.
- `requirements.txt` — список зависимостей.
- `.env` — файл с конфигурацией (необходимо создать самостоятельно).

## Использование

1. **Добавьте бота в Telegram**  
   Перейдите по ссылке [@AnimeStyleFace_bot](https://t.me/AnimeStyleFace_bot) и начните с ним диалог.

2. **Отправьте изображение**  
   Бот автоматически обработает фотографию и вернет результат в аниме-стиле.

## Лицензия

Проект распространяется под лицензией **Apache 2.0**.  
Подробности смотрите в файле [LICENSE](LICENSE).

---

**Разработано в рамках Deep Learning School.**
