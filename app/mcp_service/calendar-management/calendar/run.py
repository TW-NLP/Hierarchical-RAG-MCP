from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@mumunha/cal_dot_com_mcpserver

calcom_app = FastAPI(title="Cal.com Tools API", version="1.0.0")


# ================== Add Appointment ==================

class AddAppointmentInput(BaseModel):
    eventTypeId: int
    startTime: str  # ISO 格式：YYYY-MM-DDTHH:mm:ss.sssZ
    endTime: str    # ISO 格式：YYYY-MM-DDTHH:mm:ss.sssZ
    name: str
    email: str
    notes: Optional[str] = None

@calcom_app.post("/calcom_add_appointment", summary="Create new calendar appointments")
def calcom_add_appointment(payload: AddAppointmentInput):
    return {"success": True}


# ================== Update Appointment ==================

class UpdateAppointmentInput(BaseModel):
    bookingId: int
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    notes: Optional[str] = None

@calcom_app.post("/calcom_update_appointment", summary="Update existing calendar appointments")
def calcom_update_appointment(payload: UpdateAppointmentInput):
    return {"success": True}


# ================== Delete Appointment ==================

class DeleteAppointmentInput(BaseModel):
    bookingId: int
    reason: Optional[str] = None

@calcom_app.post("/calcom_delete_appointment", summary="Delete existing calendar appointments")
def calcom_delete_appointment(payload: DeleteAppointmentInput):
    return {"success": True}


# ================== List Appointments ==================

class ListAppointmentsInput(BaseModel):
    startDate: str  # YYYY-MM-DD
    endDate: str    # YYYY-MM-DD

@calcom_app.post("/calcom_list_appointments", summary="List calendar appointments in a date range")
def calcom_list_appointments(payload: ListAppointmentsInput):
    return {"success": True}

mcp=FastMCP.from_fastapi(app=calcom_app)
if __name__ == '__main__':
    mcp.run(transport='sse',port=50103)
    

