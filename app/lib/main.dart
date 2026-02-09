import 'package:app/screens/dashboard/dashboard_screen.dart'
    show DashboardScreen;
import 'package:flutter/material.dart';

import 'core/services/connection_config_service.dart';
import 'features/verification/services/notification_service.dart';
import 'features/verification/services/verification_tracker.dart';
import 'screens/auth/login.dart';
import 'screens/dashboard/views/home_screen.dart';
import 'screens/splash_screen.dart';
import 'screens/verification/verification_details_screen.dart';
import 'ui/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ConnectionConfigService.instance.init();
  await NotificationService.instance.init();
  // Resume tracking any verification that was in progress before the app closed.
  VerificationTracker.instance.resumeIfNeeded();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  /// Global key so we can navigate from anywhere (e.g. Dio interceptor).
  static final GlobalKey<NavigatorState> navigatorKey =
      GlobalKey<NavigatorState>();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'وثيق',
      navigatorKey: navigatorKey,
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      initialRoute: '/',
      routes: {
        '/': (context) => SplashScreen(),
        "/login": (context) => LoginScreen(),
        "/home": (context) => HomeScreen(),
        "/dashboard": (context) => DashboardScreen(),
      },
      onGenerateRoute: (settings) {
        // Handle /verification-details route (from notification tap).
        if (settings.name == '/verification-details') {
          final id = settings.arguments as int?;
          if (id != null) {
            return MaterialPageRoute(
              builder: (_) => VerificationDetailsScreen(verificationId: id),
            );
          }
        }
        return null;
      },
    );
  }
}
