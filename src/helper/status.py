from starlette.responses import JSONResponse

def handle_with_status(status, message: str | None):
    return JSONResponse(
        content={
            'message': f'Response with status {status}',
            'code': status,
            'data': {
                'message': message if message else f'Response with status {status}',
                'code': status,
            }
        },
        status_code = status
    )