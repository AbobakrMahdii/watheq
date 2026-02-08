import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../../main.dart';
import '../config/app_config.dart';
import 'auth_interceptor.dart';

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
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          options.extra['startTime'] = DateTime.now();
          debugPrint('[HTTP] --> ${options.method} ${options.uri}');
          final body = _stringifyBody(options.data);
          if (body != null && body.isNotEmpty) {
            debugPrint('[HTTP] body=$body');
          }
          handler.next(options);
        },
        onResponse: (response, handler) {
          final started = response.requestOptions.extra['startTime'] as DateTime?;
          final elapsed = started == null
              ? ''
              : ' (${DateTime.now().difference(started).inMilliseconds}ms)';
          debugPrint(
            '[HTTP] <-- ${response.statusCode} '
            '${response.requestOptions.method} '
            '${response.requestOptions.uri}$elapsed',
          );
          handler.next(response);
        },
        onError: (err, handler) {
          final req = err.requestOptions;
          final started = req.extra['startTime'] as DateTime?;
          final elapsed = started == null
              ? ''
              : ' (${DateTime.now().difference(started).inMilliseconds}ms)';
          final status = err.response?.statusCode;
          debugPrint(
            '[HTTP] <-- ERROR ${status ?? ''} ${req.method} ${req.uri}$elapsed',
          );
          final body = _stringifyBody(err.response?.data);
          if (body != null && body.isNotEmpty) {
            debugPrint('[HTTP] errorBody=$body');
          }
          handler.next(err);
        },
      ),
    );

    if (accessToken != null && accessToken.isNotEmpty) {
      _dio.options.headers['Authorization'] = 'Bearer $accessToken';

      // Only attach the 401 interceptor for authenticated requests.
      _dio.interceptors.add(AuthInterceptor(MyApp.navigatorKey));
    }
  }

  final Dio _dio;

  Dio get dio => _dio;

  String? _stringifyBody(Object? data) {
    if (data == null) {
      return null;
    }
    if (data is FormData) {
      final fieldKeys = data.fields.map((e) => e.key).toList();
      final fileKeys = data.files.map((e) => e.key).toList();
      return 'FormData(fields=${fieldKeys.join(',')}, files=${fileKeys.join(',')})';
    }
    try {
      final text = data is String ? data : jsonEncode(data);
      return text.length > 800 ? '${text.substring(0, 800)}…' : text;
    } catch (_) {
      return data.toString();
    }
  }
}

