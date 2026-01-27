import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../models/document_type_model.dart';

class DocumentTypeApiService {
  DocumentTypeApiService._();

  static final DocumentTypeApiService instance = DocumentTypeApiService._();

  Future<List<DocumentTypeModel>> getActiveDocumentTypes() async {
    try {
      final response = await ApiClient.instance.get('/api/document-types');
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => DocumentTypeModel.fromJson(json)).toList();
    } on NetworkException {
      rethrow; // Re-throw specific network exceptions
    } catch (e) {
      throw NetworkException('Failed to load document types: ${e.toString()}');
    }
  }
}
