from fastapi import APIRouter, status 
from models.predict_model import Model2B
from pydantic import BaseModel
from helper.status import handle_with_status

api_router = APIRouter(prefix='/predict')

model = Model2B()

class ModelEntity(BaseModel):
    text: str
    type: str

@api_router.post('')
async def post(data: ModelEntity):
    print(f"\033[95m=== Data in ===\033[0m")
    print(f"\033[95mType: {data.type}\033[0m")
    print(f"\033[95mText: {data.text}\033[0m")
    print(f"\033[95m=== End data in ===\033[0m")

    if len(data.text) <= 0 or len(data.type) == 0 or data is None:
        return handle_with_status(status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                  'Văn bản hoặc kích thước giới hạn phải khác 0')

    result = model.predict(data.text, data.type)

    return {
        'message': 'Done',
        'status': status.HTTP_200_OK,
        'data': result 
    }

@api_router.get('/')
async def get():
    return {
        'message': "Done",
        'status': status.HTTP_200_OK,
    }