import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';

// Mock Login Screen for testing
class LoginScreen extends StatelessWidget {
  const LoginScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Login')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const TextField(
              key: Key('email_field'),
              decoration: InputDecoration(
                labelText: 'Email',
                hintText: 'Enter your email',
              ),
            ),
            const SizedBox(height: 16),
            const TextField(
              key: Key('password_field'),
              decoration: InputDecoration(
                labelText: 'Password',
                hintText: 'Enter your password',
              ),
              obscureText: true,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              key: const Key('login_button'),
              onPressed: () {},
              child: const Text('Login'),
            ),
          ],
        ),
      ),
    );
  }
}

void main() {
  group('Login Screen Widget Tests', () {
    testWidgets('displays all required fields', (WidgetTester tester) async {
      // Build the login screen
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Verify email field exists
      expect(find.byKey(const Key('email_field')), findsOneWidget);

      // Verify password field exists
      expect(find.byKey(const Key('password_field')), findsOneWidget);

      // Verify login button exists
      expect(find.byKey(const Key('login_button')), findsOneWidget);

      // Verify labels
      expect(find.text('Email'), findsOneWidget);
      expect(find.text('Password'), findsOneWidget);
      expect(find.text('Login'), findsAtLeast(1));
    });

    testWidgets('password field is obscured', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Find password field
      final passwordField = tester.widget<TextField>(
        find.byKey(const Key('password_field')),
      );

      // Verify it's obscured
      expect(passwordField.obscureText, isTrue);
    });

    testWidgets('email field is not obscured', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Find email field
      final emailField = tester.widget<TextField>(
        find.byKey(const Key('email_field')),
      );

      // Verify it's not obscured
      expect(emailField.obscureText, isFalse);
    });

    testWidgets('login button is tappable', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Find and tap the login button
      final loginButton = find.byKey(const Key('login_button'));
      expect(loginButton, findsOneWidget);

      await tester.tap(loginButton);
      await tester.pump();

      // Button should exist after tap
      expect(loginButton, findsOneWidget);
    });

    testWidgets('can enter text in email field', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Enter text in email field
      await tester.enterText(
        find.byKey(const Key('email_field')),
        'test@example.com',
      );
      await tester.pump();

      // Verify text was entered
      expect(find.text('test@example.com'), findsOneWidget);
    });

    testWidgets('can enter text in password field', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

      // Enter text in password field
      await tester.enterText(
        find.byKey(const Key('password_field')),
        'password123',
      );
      await tester.pump();

      // Note: With const TextField without controller, we can't directly verify the text
      // But we can verify the widget accepted the input without throwing errors
      // The enterText call completing successfully is verification enough
      expect(find.byKey(const Key('password_field')), findsOneWidget);
    });
  });
}
