from fastapi import APIRouter, status

from models.predict_model import Model2B
from models.predict_model_7b import Model7B
from pydantic import BaseModel
from helper.status import handle_with_status

api_router = APIRouter(prefix="/predict")
model_type = ["2b", "7b"]

model_2b = Model2B()
model_7b = Model7B()

class ModelEntity(BaseModel):
    text: str
    type: str
    model: str | None = None


def check_match(string, array):
    return any(string == element for element in array)


@api_router.post("")
async def post(data: ModelEntity):
    print(f"\033[95m=== Data in ===\033[0m")
    print(f"\033[95mType: {data.type}\033[0m")
    print(f"\033[95mText: {data.text}\033[0m")
    print(f"\033[95m=== End data in ===\033[0m")

    model_params = '2b' if data.model is None else data.model

    if len(data.text) <= 0 or len(data.type) == 0 or data is None:
        return handle_with_status(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Văn bản hoặc kích thước giới hạn phải khác 0",
        )

    if check_match(model_params, model_type) is False:
        return handle_with_status(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Hệ thống không cung cấp mô hình này, hãy kiểm tra lại !!!",
        )

    model_selected = model_2b if model_params == '2b' else model_7b

    result = model_selected.predict(data.text, data.type)

    return {"message": "Done", "status": status.HTTP_200_OK, "data": result}


@api_router.get("/")
async def get():
    return {
        "message": "Done",
        "status": status.HTTP_200_OK,
    }
