import '../../../core/network/api_client.dart';
import '../../../features/auth/services/auth_service.dart';

/// Service for super-admin citizen records CRUD.
class CitizenService {
  CitizenService._();
  static final CitizenService instance = CitizenService._();

  Future<ApiClient> _client() async {
    final session = await AuthService.instance.getSavedSession();
    return ApiClient(accessToken: session?.accessToken);
  }

  /// Fetch paginated list of citizen records.
  Future<List<Map<String, dynamic>>> listCitizens({
    int limit = 50,
    int offset = 0,
  }) async {
    final api = await _client();
    final resp = await api.dio.get(
      '/api/v1/admin/citizens',
      queryParameters: {'limit': limit, 'offset': offset},
    );
    final data = resp.data as Map<String, dynamic>;
    final list = data['citizens'] as List<dynamic>? ?? [];
    return list.cast<Map<String, dynamic>>();
  }

  /// Get a single citizen by national ID.
  Future<Map<String, dynamic>> getCitizen(String nationalId) async {
    final api = await _client();
    final resp = await api.dio.get('/api/v1/admin/citizens/$nationalId');
    return resp.data as Map<String, dynamic>;
  }

  /// Update a citizen record.  Only non-null fields are sent.
  Future<Map<String, dynamic>> updateCitizen(
    String nationalId,
    Map<String, String?> fields,
  ) async {
    final body = <String, dynamic>{};
    fields.forEach((k, v) {
      if (v != null && v.isNotEmpty) body[k] = v;
    });
    final api = await _client();
    final resp = await api.dio.put(
      '/api/v1/admin/citizens/$nationalId',
      data: body,
    );
    return resp.data as Map<String, dynamic>;
  }

  /// Delete a citizen record.
  Future<void> deleteCitizen(String nationalId) async {
    final api = await _client();
    await api.dio.delete('/api/v1/admin/citizens/$nationalId');
  }
}
