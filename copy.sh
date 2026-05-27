#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # определили путь до скрипта
ENV_FILE="${SCRIPT_DIR}/.env" # путь для .env

if [ ! -f "$ENV_FILE" ]; then
    echo "Ошибка: Файл .env не найден в директории скрипта: $ENV_FILE" >&2
    exit 1
fi

source "$ENV_FILE"

if [ -z "$ONEFILELLM_PATH" ]; then
    echo "Ошибка: Переменная ONEFILELLM_PATH не задана в файле $ENV_FILE" >&2
    exit 1
fi

SOURCE_FILES_DIR="${SCRIPT_DIR}/src"

SOURCE_FILES=$(find "${SOURCE_FILES_DIR}" \
    -type f \
    -not -path '*/__pycache__/*' \
    -not -path '*/results/*' \
    -not -path "${SCRIPT_DIR}/static/lib/*" \
    -not -name '__pycache__' \
    -not -name '*.json')

source "${ONEFILELLM_PATH}/.venv/bin/activate" # активируем переменное окружение из папки onefilellm!
python "${ONEFILELLM_PATH}/onefilellm.py" ${SOURCE_FILES}
if [ "$REMOVE_OUTPUT" = "true" ]; then
    rm -f output.xml
    echo "Временный файл output.xml удален (локальный режим)."
else
    echo "Файл output.xml сохранен (серверный режим)."
fi
echo ${SOURCE_FILES}