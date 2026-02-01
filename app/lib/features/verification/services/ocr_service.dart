import 'dart:io';

import 'package:dio/dio.dart';

import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../auth/services/auth_service.dart';

class OcrService {
  OcrService._();

  static final OcrService instance = OcrService._();

  Future<Map<String, dynamic>> ocrDocument({
    required File file,
    int maxPages = 10,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const OcrException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final form = FormData.fromMap({
        'file': await MultipartFile.fromFile(file.path),
      });

      final response = await dio.post(
        '/api/v1/ocr',
        data: form,
        queryParameters: {'max_pages': maxPages},
      );

      final data = (response.data as Map?)?.cast<String, dynamic>() ?? {};
      return data;
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw OcrException(message, cause: e);
    }
  }
}

class OcrException implements Exception {
  const OcrException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
