import 'package:flutter/material.dart';

import 'screens/auth/login.dart';
import 'screens/home/home_screen.dart';
import 'screens/splash_screen.dart';
import 'ui/app_theme.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Watheq',
      theme: AppTheme.light(),
      initialRoute: '/',
      routes: {
        '/': (context) => SplashScreen(),
        "/login": (context) => LoginScreen(),
        "/home": (context) => HomeScreen(),
      },
    );
  }
}
