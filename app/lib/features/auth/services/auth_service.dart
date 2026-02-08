import 'package:dio/dio.dart';

import '../../../core/constants/app_keys.dart';
import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../../core/storage/secure_storage_service.dart';
import '../models/auth_session.dart';
import '../models/user_profile.dart';

class AuthService {
  AuthService._();

  static final AuthService instance = AuthService._();
  final SecureStorageService _storage = SecureStorageService.instance;

  Future<AuthSession?> getSavedSession() async {
    final accessToken = await _storage.read(AppKeys.accessToken);
    if (accessToken == null || accessToken.isEmpty) return null;

    final tokenType = await _storage.read(AppKeys.tokenType) ?? 'bearer';
    final role = await _storage.read(AppKeys.userRole) ?? 'user';
    final email = await _storage.read(AppKeys.userEmail) ?? '';

    return AuthSession(
      accessToken: accessToken,
      tokenType: tokenType,
      role: role,
      email: email,
    );
  }

  Future<void> persistSession({
    required String accessToken,
    required String tokenType,
    required String role,
    required String email,
  }) async {
    await _storage.write(AppKeys.accessToken, accessToken);
    await _storage.write(AppKeys.tokenType, tokenType);
    await _storage.write(AppKeys.userRole, role);
    await _storage.write(AppKeys.userEmail, email);
  }

  Future<void> logout() => _storage.deleteAll();

  Future<AuthSession> login({
    required String email,
    required String password,
  }) async {
    try {
      final client = ApiClient(accessToken: null).dio;
      final response = await client.post(
        '/api/v1/auth/login',
        data: FormData.fromMap({
          'username': email.trim(),
          'password': password,
        }),
        options: Options(contentType: Headers.formUrlEncodedContentType),
      );

      final data = response.data as Map<String, dynamic>;
      final accessToken = (data['access_token'] as String?) ?? '';
      final tokenType = (data['token_type'] as String?) ?? 'bearer';
      final role = (data['role'] as String?) ?? 'user';

      if (accessToken.isEmpty) {
        throw StateError('Empty access_token');
      }

      await persistSession(
        accessToken: accessToken,
        tokenType: tokenType,
        role: role,
        email: email.trim(),
      );

      return AuthSession(
        accessToken: accessToken,
        tokenType: tokenType,
        role: role,
        email: email.trim(),
      );
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw AuthException(message, cause: e);
    }
  }

  Future<bool> validateToken() async {
    final session = await getSavedSession();
    if (session == null) return false;

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      await dio.get('/api/v1/auth/me');
      return true;
    } catch (_) {
      await logout();
      return false;
    }
  }

  Future<UserProfile> fetchProfile() async {
    final session = await getSavedSession();
    if (session == null) {
      throw AuthException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final res = await dio.get('/api/v1/auth/me');
      return UserProfile.fromJson(res.data as Map<String, dynamic>);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw AuthException(message, cause: e);
    }
  }

  Future<void> changePassword({
    required String currentPassword,
    required String newPassword,
  }) async {
    final session = await getSavedSession();
    if (session == null) {
      throw AuthException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      await dio.put(
        '/api/v1/auth/change-password',
        data: {
          'current_password': currentPassword,
          'new_password': newPassword,
        },
      );
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw AuthException(message, cause: e);
    }
  }
}

class AuthException implements Exception {
  const AuthException(this.message, {this.cause});

  final String message;
  final Object? cause;
}
