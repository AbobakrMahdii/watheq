from typing import List
from fastapi import APIRouter, Depends
from api.services.document_type_service import DocumentTypeService, get_document_type_service
from api.models import DocumentTypePublic

router = APIRouter(
    prefix="/api/document-types",
    tags=["Document Types (Public)"]
)

@router.get(
    "", 
    response_model=List[DocumentTypePublic],
    summary="Get all active document types",
    description="Retrieves a list of all active document types. No authentication required."
)
async def get_active_document_types(
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Get all active document types."""
    return await service.get_active_document_types()
