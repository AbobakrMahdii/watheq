from typing import List, Dict, Any, Optional
from api.database import DocumentTypesCollection, get_document_type_collection
from api.models import DocumentTypeCreate, DocumentTypeUpdate, DocumentTypeInDB
from fastapi import Depends # Add this import

class DocumentTypeService:
    def __init__(self, collection: DocumentTypesCollection = None):
        self.collection = collection or get_document_type_collection()

    async def get_active_document_types(self) -> List[DocumentTypeInDB]:
        """Returns only active document types."""
        raw_docs = await self.collection.find(filt={"is_active": True})
        return [DocumentTypeInDB.model_validate(doc) for doc in raw_docs]

    async def get_all_document_types(self) -> List[DocumentTypeInDB]:
        """Returns all document types, active or inactive."""
        raw_docs = await self.collection.find()
        return [DocumentTypeInDB.model_validate(doc) for doc in raw_docs]

    async def get_document_type_by_id(self, doc_id: int) -> Optional[DocumentTypeInDB]:
        """Returns a single document type by ID."""
        raw_doc = await self.collection.find_one(filt={"_id": doc_id})
        return DocumentTypeInDB.model_validate(raw_doc) if raw_doc else None

    async def create_document_type(self, doc_type_data: DocumentTypeCreate) -> DocumentTypeInDB:
        """Creates a new document type."""
        doc_dict = doc_type_data.model_dump()

        # Insert into DB, get the new ID (insert_one returns the last inserted ID)
        new_id = await self.collection.insert_one(doc_dict)
        
        # Fetch the newly created document type to get its `created_at` and `id` from the DB
        created_doc = await self.collection.find_one(filt={"_id": new_id})
        if not created_doc:
            raise ValueError("Failed to retrieve newly created document type")
        return DocumentTypeInDB.model_validate(created_doc)

    async def update_document_type(self, doc_id: int, update_data: DocumentTypeUpdate) -> Optional[DocumentTypeInDB]:
        """Updates an existing document type."""
        update_dict = {k: v for k, v in update_data.model_dump(exclude_unset=True).items() if v is not None}
        if not update_dict:
            return await self.get_document_type_by_id(doc_id) # No update needed

        rows_affected = await self.collection.update_one(doc_id, update_dict)
        if rows_affected > 0:
            return await self.get_document_type_by_id(doc_id)
        return None

    async def delete_document_type(self, doc_id: int) -> bool:
        """Deletes a document type by ID."""
        rows_affected = await self.collection.delete_one(doc_id)
        return rows_affected > 0

# Dependency for FastAPI
def get_document_type_service(collection: DocumentTypesCollection = Depends(get_document_type_collection)):
    return DocumentTypeService(collection)
