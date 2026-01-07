import 'dart:io';

import 'package:dio/dio.dart';

import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../auth/services/auth_service.dart';
import '../models/ipfs_pin_result.dart';

class IpfsService {
  IpfsService._();

  static final IpfsService instance = IpfsService._();

  Future<IpfsPinResult> pinFile({required File file}) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const IpfsException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final form = FormData.fromMap({
        'file': await MultipartFile.fromFile(file.path),
      });
      final response = await dio.post('/api/v1/ipfs/pin-file', data: form);
      final data = (response.data as Map?)?.cast<String, dynamic>() ?? {};
      return IpfsPinResult.fromJson(data);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw IpfsException(message, cause: e);
    }
  }
}

class IpfsException implements Exception {
  const IpfsException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
