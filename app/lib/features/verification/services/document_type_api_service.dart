import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../models/document_type_model.dart';

class DocumentTypeApiService {
  DocumentTypeApiService._();

  static final DocumentTypeApiService instance = DocumentTypeApiService._();

  Future<List<DocumentTypeModel>> getActiveDocumentTypes() async {
    try {
      // Public endpoint, no auth token required.
      final dio = ApiClient(accessToken: null).dio;
      final response = await dio.get('/api/document-types');
      final data = response.data;
      final list = (data is List) ? data : const [];
      return list.map((json) => DocumentTypeModel.fromJson(json)).toList();
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw Exception(message);
    }
  }
}
