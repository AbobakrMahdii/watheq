"use client";

import React, { useEffect, useState } from "react";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

// Assuming these models exist in the backend types or are defined locally
type DocumentType = {
  id: number;
  name: string;
  is_active: boolean;
  requires_back_image: boolean;
  created_at: string;
};

type DocumentTypeCreatePayload = {
  name: string;
  is_active?: boolean;
  requires_back_image?: boolean;
};

type DocumentTypeUpdatePayload = {
  name?: string;
  is_active?: boolean;
  requires_back_image?: boolean;
};

export default function DocumentTypesPage() {
  const router = useRouter();
  const [documentTypes, setDocumentTypes] = useState<DocumentType[]>([]);
  const [loading, setLoading] = useState(false);
  const [formName, setFormName] = useState("");
  const [formIsActive, setFormIsActive] = useState(true);
  const [formRequiresBackImage, setFormRequiresBackImage] = useState(false);
  const [editingDocTypeId, setEditingDocTypeId] = useState<number | null>(null);

  async function loadDocumentTypes() {
    setLoading(true);
    try {
      const res = await fetch("/api/admin/document-types");
      const data = await res.json().catch(() => []);
      if (!res.ok) throw new Error(data?.message || "Failed to load document types");
      setDocumentTypes(Array.isArray(data) ? data : []);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load document types");
      // router.push("/auth/login"); // Redirect to login if unauthorized
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDocumentTypes();
  }, []);

  function resetForm() {
    setFormName("");
    setFormIsActive(true);
    setFormRequiresBackImage(false);
    setEditingDocTypeId(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!formName.trim()) {
      return toast.error("Document type name is required");
    }

    setLoading(true);
    try {
      let res: Response;
      let payload: DocumentTypeCreatePayload | DocumentTypeUpdatePayload = {
        name: formName.trim(),
        is_active: formIsActive,
        requires_back_image: formRequiresBackImage,
      };

      if (editingDocTypeId) {
        // Update existing document type
        res = await fetch(`/api/admin/document-types/${editingDocTypeId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } else {
        // Create new document type
        res = await fetch("/api/admin/document-types", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || (editingDocTypeId ? "Failed to update document type" : "Failed to create document type"));

      toast.success(editingDocTypeId ? "Document type updated" : "Document type created");
      resetForm();
      await loadDocumentTypes(); // Reload the list
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function toggleActive(id: number, currentStatus: boolean) {
    setLoading(true);
    try {
      const res = await fetch(`/api/admin/document-types/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_active: !currentStatus }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Failed to toggle status");

      toast.success("Status updated");
      await loadDocumentTypes();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function deleteDocumentType(id: number) {
    if (!confirm("Are you sure you want to delete this document type?")) return;

    setLoading(true);
    try {
      const res = await fetch(`/api/admin/document-types/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete document type");

      toast.success("Document type deleted");
      await loadDocumentTypes();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function startEditing(docType: DocumentType) {
    setEditingDocTypeId(docType.id);
    setFormName(docType.name);
    setFormIsActive(docType.is_active);
    setFormRequiresBackImage(docType.requires_back_image);
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <header className="bg-white shadow p-4 mb-4">
        <h1 className="text-2xl font-semibold">Manage Document Types</h1>
      </header>

      <main className="flex-1 p-6">
        <div className="bg-white p-4 rounded shadow mb-6">
          <h2 className="text-xl font-semibold mb-3">{editingDocTypeId ? "Edit Document Type" : "Add New Document Type"}</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
                required
                disabled={loading}
              />
            </div>
            <div className="flex items-center gap-4 mt-6 md:mt-1">
              <label className="flex items-center text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={formIsActive}
                  onChange={(e) => setFormIsActive(e.target.checked)}
                  className="form-checkbox h-4 w-4 text-blue-600 border-slate-300 rounded"
                  disabled={loading}
                />
                <span className="ml-2">Is Active</span>
              </label>
              <label className="flex items-center text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={formRequiresBackImage}
                  onChange={(e) => setFormRequiresBackImage(e.target.checked)}
                  className="form-checkbox h-4 w-4 text-blue-600 border-slate-300 rounded"
                  disabled={loading}
                />
                <span className="ml-2">Requires Back Image</span>
              </label>
            </div>
            <div className="md:col-span-2 flex justify-end gap-2">
              <button
                type="submit"
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
                disabled={loading}
              >
                {editingDocTypeId ? "Update Document Type" : "Add Document Type"}
              </button>
              {editingDocTypeId && (
                <button
                  type="button"
                  onClick={resetForm}
                  className="bg-slate-300 text-slate-800 px-4 py-2 rounded-md hover:bg-slate-400 disabled:opacity-50"
                  disabled={loading}
                >
                  Cancel Edit
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-semibold mb-3">Existing Document Types</h2>
          {documentTypes.length === 0 && !loading ? (
            <p className="text-slate-500">No document types found. Add one above!</p>
          ) : (
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">ID</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Name</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Active</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Back Image</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Created At</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-slate-200">
                {documentTypes.map((docType) => (
                  <tr key={docType.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{docType.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{docType.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${docType.is_active ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                        {docType.is_active ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${docType.requires_back_image ? "bg-blue-100 text-blue-800" : "bg-slate-100 text-slate-800"}`}>
                        {docType.requires_back_image ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{new Date(docType.created_at).toLocaleDateString()}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => startEditing(docType)}
                        className="text-indigo-600 hover:text-indigo-900 mr-4"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => toggleActive(docType.id, docType.is_active)}
                        className={`${docType.is_active ? "text-red-600 hover:text-red-900" : "text-green-600 hover:text-green-900"} mr-4`}
                        disabled={loading}
                      >
                        {docType.is_active ? "Deactivate" : "Activate"}
                      </button>
                      <button
                        onClick={() => deleteDocumentType(docType.id)}
                        className="text-red-600 hover:text-red-900"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
