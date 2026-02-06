import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';

import '../../auth/services/auth_service.dart';
import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../models/verification_models.dart';

class VerificationOrchestratorService {
  VerificationOrchestratorService._();

  static final VerificationOrchestratorService instance =
      VerificationOrchestratorService._();

  Future<VerificationRecord> start({
    required File documentImageFront,
    File? documentImageBack,
    required File personImage,
    required int documentTypeId,
    Map<String, dynamic>? livenessData,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const VerificationOrchestratorException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final form = FormData();
      form.fields.add(MapEntry('document_type_id', documentTypeId.toString()));
      form.files.add(
        MapEntry(
          'document_image_front',
          await MultipartFile.fromFile(documentImageFront.path),
        ),
      );
      form.files.add(
        MapEntry(
          'person_image',
          await MultipartFile.fromFile(personImage.path),
        ),
      );
      if (documentImageBack != null) {
        form.files.add(
          MapEntry(
            'document_image_back',
            await MultipartFile.fromFile(documentImageBack.path),
          ),
        );
      }
      if (livenessData != null) {
        form.fields.add(MapEntry('liveness_data', jsonEncode(livenessData)));
      }

      final response = await dio.post('/api/v1/verifications/start', data: form);
      return VerificationRecord.fromJson(response.data as Map<String, dynamic>);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw VerificationOrchestratorException(message, cause: e);
    }
  }

  Future<VerificationRecord> getStatus(int verificationId) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const VerificationOrchestratorException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final response = await dio.get('/api/v1/verifications/$verificationId');
      return VerificationRecord.fromJson(response.data as Map<String, dynamic>);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw VerificationOrchestratorException(message, cause: e);
    }
  }

  Future<List<VerificationStep>> getSteps(int verificationId) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const VerificationOrchestratorException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final response = await dio.get('/api/v1/verifications/$verificationId/steps');
      final data = response.data as List<dynamic>;
      return data
          .map((e) => VerificationStep.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw VerificationOrchestratorException(message, cause: e);
    }
  }

  Future<VerificationSummary> listMy({
    int page = 1,
    int pageSize = 10,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const VerificationOrchestratorException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      final response = await dio.get(
        '/api/v1/verifications/my',
        queryParameters: {'page': page, 'page_size': pageSize},
      );
      return VerificationSummary.fromJson(response.data as Map<String, dynamic>);
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw VerificationOrchestratorException(message, cause: e);
    }
  }
}

class VerificationOrchestratorException implements Exception {
  const VerificationOrchestratorException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
