"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { SortableHeader } from "@/components/sortable-header";
import { useSortState } from "@/hooks/use-sort-state";
import { sortLocally } from "@/lib/sort-utils";

// Assuming these models exist in the backend types or are defined locally
type DocumentType = {
  id: number;
  name: string;
  folder_name: string;
  is_active: boolean;
  requires_back_image: boolean;
  created_at: string;
};

type DocumentTypeCreatePayload = {
  name: string;
  folder_name: string;
  is_active?: boolean;
  requires_back_image?: boolean;
};

type DocumentTypeUpdatePayload = {
  name?: string;
  folder_name?: string;
  is_active?: boolean;
  requires_back_image?: boolean;
};

export default function DocumentTypesPage() {
  const router = useRouter();
  const [documentTypes, setDocumentTypes] = useState<DocumentType[]>([]);
  const [loading, setLoading] = useState(false);
  const [formName, setFormName] = useState("");
  const [formFolderName, setFormFolderName] = useState("");
  const [formIsActive, setFormIsActive] = useState(true);
  const [formRequiresBackImage, setFormRequiresBackImage] = useState(false);
  const [editingDocTypeId, setEditingDocTypeId] = useState<number | null>(null);
  const { sortBy, sortOrder, toggleSort } = useSortState("id", "asc");

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
    setFormFolderName("");
    setFormIsActive(true);
    setFormRequiresBackImage(false);
    setEditingDocTypeId(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!formName.trim()) {
      return toast.error("Document type name is required");
    }
    if (!formFolderName.trim()) {
      return toast.error("Folder name is required");
    }

    setLoading(true);
    try {
      let res: Response;
      let payload: DocumentTypeCreatePayload | DocumentTypeUpdatePayload = {
        name: formName.trim(),
        folder_name: formFolderName.trim(),
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
      if (!res.ok)
        throw new Error(
          data?.message || (editingDocTypeId ? "Failed to update document type" : "Failed to create document type"),
        );

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
    setFormFolderName(docType.folder_name);
    setFormIsActive(docType.is_active);
    setFormRequiresBackImage(docType.requires_back_image);
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <header className="mb-4 flex items-center justify-between bg-white p-4 shadow">
        <h1 className="text-2xl font-semibold">Manage Document Types</h1>
        {/* Quick access back to the main admin dashboard. */}
        <Link href="/dashboard" className="text-sm text-slate-600 hover:text-slate-900">
          Back to Dashboard
        </Link>
      </header>

      <main className="flex-1 p-6">
        <div className="mb-6 rounded bg-white p-4 shadow">
          <h2 className="mb-3 text-xl font-semibold">
            {editingDocTypeId ? "Edit Document Type" : "Add New Document Type"}
          </h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-slate-700">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="mt-1 block w-full rounded-md border border-slate-300 p-2 shadow-sm"
                required
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">Folder Name</label>
              <input
                type="text"
                value={formFolderName}
                onChange={(e) => setFormFolderName(e.target.value)}
                className="mt-1 block w-full rounded-md border border-slate-300 p-2 shadow-sm"
                placeholder="e.g., identity, passport"
                required
                disabled={loading}
              />
            </div>
            <div className="mt-6 flex items-center gap-4 md:mt-1">
              <label className="flex items-center text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={formIsActive}
                  onChange={(e) => setFormIsActive(e.target.checked)}
                  className="form-checkbox h-4 w-4 rounded border-slate-300 text-blue-600"
                  disabled={loading}
                />
                <span className="ml-2">Is Active</span>
              </label>
              <label className="flex items-center text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={formRequiresBackImage}
                  onChange={(e) => setFormRequiresBackImage(e.target.checked)}
                  className="form-checkbox h-4 w-4 rounded border-slate-300 text-blue-600"
                  disabled={loading}
                />
                <span className="ml-2">Requires Back Image</span>
              </label>
            </div>
            <div className="flex justify-end gap-2 md:col-span-2">
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
                disabled={loading}
              >
                {editingDocTypeId ? "Update Document Type" : "Add Document Type"}
              </button>
              {editingDocTypeId && (
                <button
                  type="button"
                  onClick={resetForm}
                  className="rounded-md bg-slate-300 px-4 py-2 text-slate-800 hover:bg-slate-400 disabled:opacity-50"
                  disabled={loading}
                >
                  Cancel Edit
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="rounded bg-white p-4 shadow">
          <h2 className="mb-3 text-xl font-semibold">Existing Document Types</h2>
          {documentTypes.length === 0 && !loading ? (
            <p className="text-slate-500">No document types found. Add one above!</p>
          ) : (
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    <SortableHeader
                      label="ID"
                      field="id"
                      currentSortBy={sortBy}
                      currentSortOrder={sortOrder}
                      onSort={toggleSort}
                      className="text-xs text-slate-500 uppercase"
                    />
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    <SortableHeader
                      label="Name"
                      field="name"
                      currentSortBy={sortBy}
                      currentSortOrder={sortOrder}
                      onSort={toggleSort}
                      className="text-xs text-slate-500 uppercase"
                    />
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    <SortableHeader
                      label="Folder"
                      field="folder_name"
                      currentSortBy={sortBy}
                      currentSortOrder={sortOrder}
                      onSort={toggleSort}
                      className="text-xs text-slate-500 uppercase"
                    />
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    <SortableHeader
                      label="Active"
                      field="is_active"
                      currentSortBy={sortBy}
                      currentSortOrder={sortOrder}
                      onSort={toggleSort}
                      className="text-xs text-slate-500 uppercase"
                    />
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    Back Image
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    <SortableHeader
                      label="Created At"
                      field="created_at"
                      currentSortBy={sortBy}
                      currentSortOrder={sortOrder}
                      onSort={toggleSort}
                      className="text-xs text-slate-500 uppercase"
                    />
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-slate-500 uppercase"
                  >
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 bg-white">
                {sortLocally(documentTypes, sortBy, sortOrder).map((docType) => (
                  <tr key={docType.id}>
                    <td className="px-6 py-4 text-sm font-medium whitespace-nowrap text-slate-900">{docType.id}</td>
                    <td className="px-6 py-4 text-sm whitespace-nowrap text-slate-500">{docType.name}</td>
                    <td className="px-6 py-4 text-sm whitespace-nowrap text-slate-500">
                      <code className="rounded bg-slate-100 px-2 py-1 text-xs">{docType.folder_name}</code>
                    </td>
                    <td className="px-6 py-4 text-sm whitespace-nowrap text-slate-500">
                      <span
                        className={`inline-flex rounded-full px-2 text-xs leading-5 font-semibold ${docType.is_active ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}
                      >
                        {docType.is_active ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm whitespace-nowrap text-slate-500">
                      <span
                        className={`inline-flex rounded-full px-2 text-xs leading-5 font-semibold ${docType.requires_back_image ? "bg-blue-100 text-blue-800" : "bg-slate-100 text-slate-800"}`}
                      >
                        {docType.requires_back_image ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm whitespace-nowrap text-slate-500">
                      {new Date(docType.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-right text-sm font-medium whitespace-nowrap">
                      <button
                        onClick={() => startEditing(docType)}
                        className="mr-4 text-indigo-600 hover:text-indigo-900"
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
