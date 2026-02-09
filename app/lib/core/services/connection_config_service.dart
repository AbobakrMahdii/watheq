import 'package:shared_preferences/shared_preferences.dart';

/// Manages the backend connection IP & port.
///
/// Values are cached to SharedPreferences only after a successful login.
/// On next launch the cached values are loaded as defaults.
class ConnectionConfigService {
  ConnectionConfigService._();

  static final ConnectionConfigService instance = ConnectionConfigService._();

  static const String _keyIp = 'connection_ip';
  static const String _keyPort = 'connection_port';
  static const String _keyDashboardPort = 'dashboard_port';

  // Defaults – used when nothing is cached yet.
  static const String defaultIp = '192.168.8.36';
  static const String defaultPort = '8012';
  static const String defaultDashboardPort = '3200';

  String _ip = defaultIp;
  String _port = defaultPort;
  String _dashboardPort = defaultDashboardPort;

  String get ip => _ip;
  String get port => _port;
  String get dashboardPort => _dashboardPort;

  /// The fully constructed base URL: `http://<ip>:<port>`
  String get baseUrl => 'http://$_ip:$_port';

  /// The fully constructed dashboard URL: `http://<ip>:<dashboardPort>`
  String get dashboardUrl => 'http://$_ip:$_dashboardPort';

  /// Call once at app startup to hydrate from cache.
  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _ip = prefs.getString(_keyIp) ?? defaultIp;
    _port = prefs.getString(_keyPort) ?? defaultPort;
    _dashboardPort = prefs.getString(_keyDashboardPort) ?? defaultDashboardPort;
  }

  /// Updates the in-memory values (does NOT persist yet).
  void setConnection({
    required String ip,
    required String port,
    String? dashboardPort,
  }) {
    _ip = ip.trim();
    _port = port.trim();
    if (dashboardPort != null) {
      _dashboardPort = dashboardPort.trim();
    }
  }

  /// Persists current values to SharedPreferences.
  /// Call this only after a **successful** login.
  Future<void> persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyIp, _ip);
    await prefs.setString(_keyPort, _port);
    await prefs.setString(_keyDashboardPort, _dashboardPort);
  }
}
