from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Response
from api.services.document_type_service import DocumentTypeService, get_document_type_service
from api.models import DocumentTypeCreate, DocumentTypeUpdate, DocumentTypePublic, DocumentTypeInDB
from api.security import get_current_admin # Requires Admin role for access

router = APIRouter(
    prefix="/api/admin/document-types",
    tags=["Admin - Document Types"],
    dependencies=[Depends(get_current_admin)] # All endpoints in this router require admin access
)

@router.get(
    "", 
    response_model=List[DocumentTypePublic],
    summary="Get all document types (Admin only)",
    description="Retrieves a list of all document types, including inactive ones. Requires Admin authentication."
)
async def get_all_document_types(
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Get all document types."""
    return await service.get_all_document_types()

@router.post(
    "", 
    response_model=DocumentTypePublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new document type (Admin only)",
    description="Creates a new document type. Requires Admin authentication."
)
async def create_document_type(
    doc_type_data: DocumentTypeCreate,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Create a new document type."""
    # Check if a document type with the same name already exists
    existing_doc_type = await service.collection.find_one({"name": doc_type_data.name})
    if existing_doc_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document type with this name already exists")

    created_doc_type = await service.create_document_type(doc_type_data)
    return created_doc_type

@router.get(
    "/{doc_type_id}", 
    response_model=DocumentTypePublic,
    summary="Get a document type by ID (Admin only)",
    description="Retrieves a single document type by its ID. Requires Admin authentication."
)
async def get_document_type(
    doc_type_id: int,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Get a document type by ID."""
    doc_type = await service.get_document_type_by_id(doc_type_id)
    if not doc_type:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document type not found")
    return doc_type

@router.put(
    "/{doc_type_id}", 
    response_model=DocumentTypePublic,
    summary="Update a document type by ID (Admin only)",
    description="Updates an existing document type by its ID. Requires Admin authentication."
)
async def update_document_type(
    doc_type_id: int,
    update_data: DocumentTypeUpdate,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Update a document type by ID."""
    updated_doc_type = await service.update_document_type(doc_type_id, update_data)
    if not updated_doc_type:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document type not found or no changes applied")
    return updated_doc_type

@router.delete(
    "/{doc_type_id}", 
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document type by ID (Admin only)",
    description="Deletes a document type by its ID. Requires Admin authentication."
)
async def delete_document_type(
    doc_type_id: int,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Delete a document type by ID."""
    success = await service.delete_document_type(doc_type_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document type not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
