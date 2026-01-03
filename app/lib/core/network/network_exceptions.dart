import 'package:dio/dio.dart';

class NetworkExceptions {
  const NetworkExceptions._();

  static String toUserMessage(Object error) {
    if (error is DioException) {
      if (error.type == DioExceptionType.connectionTimeout ||
          error.type == DioExceptionType.sendTimeout ||
          error.type == DioExceptionType.receiveTimeout) {
        return 'تعذر الاتصال بالخادم. حاول مرة أخرى.';
      }

      final response = error.response;
      if (response != null) {
        final data = response.data;
        final detail = _extractDetail(data);
        if (detail != null && detail.trim().isNotEmpty) return detail;
        return 'حدث خطأ من الخادم (${response.statusCode}).';
      }

      return 'تعذر تنفيذ الطلب. تحقق من الاتصال.';
    }

    return 'حدث خطأ غير متوقع.';
  }

  static String? _extractDetail(dynamic data) {
    if (data is Map<String, dynamic>) {
      final detail = data['detail'];
      if (detail is String) return detail;
      if (detail is List && detail.isNotEmpty) {
        final first = detail.first;
        if (first is Map<String, dynamic> && first['msg'] is String) {
          return first['msg'] as String;
        }
      }
    }
    return null;
  }
}

