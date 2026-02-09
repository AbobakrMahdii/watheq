import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

import '../../features/auth/services/auth_service.dart';

/// Dio interceptor that catches 401 responses, clears the stored
/// session, and navigates to the login screen.
class AuthInterceptor extends Interceptor {
  AuthInterceptor(this._navigatorKey);

  final GlobalKey<NavigatorState> _navigatorKey;

  /// Prevent multiple concurrent redirects.
  static bool _redirecting = false;

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    if (err.response?.statusCode == 401 && !_redirecting) {
      _redirecting = true;

      try {
        // Clear stored tokens.
        await AuthService.instance.logout();

        // Navigate to login, clearing the back stack.
        _navigatorKey.currentState?.pushNamedAndRemoveUntil(
          '/login',
          (_) => false,
        );
      } finally {
        // Reset after a short delay so subsequent 401s (from parallel
        // requests) don't each trigger a redirect.
        Future.delayed(const Duration(seconds: 2), () {
          _redirecting = false;
        });
      }
    }

    handler.next(err);
  }
}
