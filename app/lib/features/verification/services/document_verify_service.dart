import 'dart:io';

import 'package:dio/dio.dart';

import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../auth/services/auth_service.dart';
import '../models/document_verify_result.dart';

class DocumentVerifyService {
  DocumentVerifyService._();

  static final DocumentVerifyService instance = DocumentVerifyService._();

  Future<DocumentVerifyResult> verify({required File documentImage}) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const DocumentVerifyException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final form = FormData.fromMap({
        'file': await MultipartFile.fromFile(documentImage.path),
      });
      final response = await dio.post('/api/v1/document/verify', data: form);
      final data = (response.data as Map?)?.cast<String, dynamic>() ?? {};
      return DocumentVerifyResult.fromJson(data);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw DocumentVerifyException(message, cause: e);
    }
  }
}

class DocumentVerifyException implements Exception {
  const DocumentVerifyException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}

