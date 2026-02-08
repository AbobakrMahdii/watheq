class AppConfig {
  const AppConfig._();

  // Backend base URL reachable من الجهاز (استخدم عنوان الشبكة Wi-Fi)
  // تأكد أن الـ Backend يعمل على 0.0.0.0:8012
  static const String apiBaseUrl = 'http://192.168.8.36:8012';
}
