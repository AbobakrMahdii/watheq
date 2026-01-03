import 'package:dio/dio.dart';

import '../config/app_config.dart';

class ApiClient {
  ApiClient({
    required String? accessToken,
  }) : _dio = Dio(
          BaseOptions(
            baseUrl: AppConfig.apiBaseUrl,
            connectTimeout: const Duration(seconds: 20),
            receiveTimeout: const Duration(seconds: 20),
            headers: const {'Accept': 'application/json'},
          ),
        ) {
    if (accessToken != null && accessToken.isNotEmpty) {
      _dio.options.headers['Authorization'] = 'Bearer $accessToken';
    }
  }

  final Dio _dio;

  Dio get dio => _dio;
}

