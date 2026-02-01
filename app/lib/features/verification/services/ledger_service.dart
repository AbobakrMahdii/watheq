import '../../../core/network/api_client.dart';
import '../../../core/network/network_exceptions.dart';
import '../../auth/services/auth_service.dart';

class LedgerService {
  LedgerService._();

  static final LedgerService instance = LedgerService._();

  Future<void> createDoc({
    required String docId,
    required String cid,
    required String filename,
    required String owner,
    required String sha256,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const LedgerException('غير مسجل الدخول');
    }

    try {
      final dio = ApiClient(accessToken: session.accessToken).dio;
      await dio.post(
        '/api/v1/ledger/docs',
        data: {
          'doc_id': docId,
          'cid': cid,
          'filename': filename,
          'owner': owner,
          'sha256': sha256,
        },
      );
    } catch (e) {
      final message = NetworkExceptions.toUserMessage(e);
      throw LedgerException(message, cause: e);
    }
  }
}

class LedgerException implements Exception {
  const LedgerException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
