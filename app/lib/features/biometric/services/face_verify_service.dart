import 'dart:io';

import 'package:dio/dio.dart';

import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../auth/services/auth_service.dart';
import '../models/face_verify_result.dart';

class FaceVerifyService {
  FaceVerifyService._();

  static final FaceVerifyService instance = FaceVerifyService._();

  Future<FaceVerifyResult> verify({
    required File documentPhoto,
    required File personPhoto,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const FaceVerifyException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final form = FormData.fromMap({
        'photo1': await MultipartFile.fromFile(documentPhoto.path),
        'photo2': await MultipartFile.fromFile(personPhoto.path),
      });

      final response = await dio.post('/api/v1/face/verify', data: form);
      final data = (response.data as Map?)?.cast<String, dynamic>() ?? {};
      return FaceVerifyResult.fromJson(data);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw FaceVerifyException(message, cause: e);
    }
  }
}

class FaceVerifyException implements Exception {
  const FaceVerifyException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
