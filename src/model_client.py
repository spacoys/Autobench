import truststore
truststore.inject_into_ssl()

import os
import asyncio
import logging
import httpx
from openai import AsyncOpenAI
from typing import Optional, Dict

# Логирование для отладки сетевых запросов
logger = logging.getLogger(__name__)

# Глобальный семафор: не более 1 запроса одновременно.
# Для Groq free tier это критично — иначе rate limit и Connection error.
_semaphore = asyncio.Semaphore(1)


class ModelClient:
    def __init__(self, config: Dict):
        # --- Читаем настройки из конфига или переменных окружения ---
        self.base_url = config.get(
            'base_url',
            os.getenv('OPENAI_BASE_URL', 'https://api.groq.com/openai/v1')
        )
        self.api_key = config.get(
            'api_key',
            os.getenv('OPENAI_API_KEY', 'dummy-key')
        )
        self.model_name = config['name']
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1024)

        # --- Клиент с явными таймаутами и повторами ---
        self.client = AsyncOpenAI(
    api_key=self.api_key,
    base_url=self.base_url,
    timeout=httpx.Timeout(60.0, connect=15.0),
    max_retries=0
)

        logger.info(
            f"ModelClient инициализирован: model={self.model_name}, "
            f"base_url={self.base_url}, max_tokens={self.max_tokens}"
        )

    async def ask(self, question: str, context: Optional[str] = None) -> str:
        """
        Отправляет запрос к модели.
        Если context задан — работает в RAG-режиме, иначе zero-shot.
        """
        messages = []

        if context:
            messages.append({
                "role": "system",
                "content": (
                    "Ты — полезный ассистент. Отвечай СТРОГО на русском языке, "
                    "кратко и по существу, используя предоставленный контекст. "
                    "Если ответа нет в контексте, скажи об этом. "
                    "Не добавляй рассуждения и пояснения — только финальный ответ."
                )
            })
            messages.append({
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос: {question}"
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "Ты — полезный ассистент. Отвечай СТРОГО на русском языке, "
                    "кратко и по существу."
                )
            })
            messages.append({"role": "user", "content": question})

        # Ограничиваем параллелизм: не более 1 одновременного запроса
        async with _semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Извлекаем контент
                if not response.choices:
                    return "[EMPTY] Нет choices в ответе"

                content = response.choices[0].message.content
                if not content:
                    return "[EMPTY] Пустой content"

                return content.strip()

            except Exception as e:
                # Возвращаем тип ошибки, чтобы её было видно в JSON
                err_type = type(e).__name__
                err_msg = str(e)
                logger.error(f"Ошибка при запросе к модели: {err_type}: {err_msg}")
                return f"[ERROR] {err_type}: {err_msg}"


# --- Опционально: включение DEBUG-логирования для httpx ---
# Раскомментируйте, если нужно видеть детали сетевых запросов
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)