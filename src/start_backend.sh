echo "\033[92mStart Backend\033[0m"
uvicorn main:app --host 0.0.0.0 --port 5005 --reload --access-log