import os
import logging
import asyncio
from io import BytesIO

import torch
from PIL import Image
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.types import BufferedInputFile
from aiogram.filters import Command
from torchvision import transforms
from model import Generator

API_TOKEN = '7603984249:AAFPfVGKRYr1FB-7MmC8PmKqqBV8HmKHnuU'
MODEL_PATH = '/home/serttyzar/Projects/Anime_styling/models/cyclegan_128_135.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

dp.include_router(router)

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    model = Generator().to(DEVICE)
    model.load_state_dict(checkpoint['G_real_to_anime'])
    return model

style_transfer_model = load_model()


def process_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output_tensor = style_transfer_model(image_tensor).squeeze(0)
        
        output_tensor = output_tensor * 0.5 + 0.5
        output_tensor = output_tensor.clamp(0, 1)
        
        to_pil = transforms.ToPILImage()
        output_image = to_pil(output_tensor.cpu())
        
        return output_image
        
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return None


@router.message(Command('start'))
@router.message(Command('help'))
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне фото лица в анфас, и я преобразую его в аниме-стиль!")


@router.message(F.photo)
async def handle_photo(message: types.Message):
    try:
        processing_msg = await message.reply("🔄 Обрабатываю изображение...")
        
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        image_bytes = await bot.download_file(file.file_path)
        
        output_image = await asyncio.to_thread(
            process_image, 
            image_bytes.read()
        )
        
        if output_image:
            output_bytes = BytesIO()
            output_image.save(output_bytes, format='JPEG')
            output_bytes.seek(0)
            
            await message.reply_photo(
                BufferedInputFile(output_bytes.getvalue(), filename='result.jpg'),
                caption="Готово! Ваше изображение в аниме-стиле 🎨"
            )
        else:
            await message.reply("❌ Произошла ошибка при обработке изображения")
        
    except Exception as e:
        logging.error(f"Handler error: {e}")
        await message.reply("❌ Произошла ошибка при обработке запроса")
    
    finally:
        await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())