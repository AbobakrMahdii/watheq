import '../services/connection_config_service.dart';

class AppConfig {
  const AppConfig._();

  /// Dynamic backend base URL constructed from cached IP & port.
  /// Updated at runtime via the login screen connection settings.
  static String get apiBaseUrl => ConnectionConfigService.instance.baseUrl;
}
