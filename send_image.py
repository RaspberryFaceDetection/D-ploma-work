import os

import telegram

from settings import settings


def send_image_to_telegram(image, name, identification_time):
    message = f"{name} пройшов ідентифікацію о {identification_time}"
    bot = telegram.Bot(settings.CONFIG["TELEGRAM_TOKEN"])
    mes = bot.send_photo(settings.CONFIG["TELEGRAM_CHAT"], image, message)
    return mes.text


if __name__ == "__main__":
    for file in os.listdir("faces"):
        filename, file_extension = os.path.splitext(file)
        person_name = filename.split("_")[0]
        identification_time = filename.split("_")[1]
        send_image_to_telegram(
            open(f"faces/{file}", "rb"), person_name, identification_time
        )
        os.remove(f"faces/{file}")
