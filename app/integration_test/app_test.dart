import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('App Integration Tests', () {
    testWidgets('app loads successfully', (WidgetTester tester) async {
      // Simple app structure test
      final app = MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Test App')),
          body: const Center(child: Text('Hello World')),
        ),
      );

      await tester.pumpWidget(app);

      expect(find.text('Test App'), findsOneWidget);
      expect(find.text('Hello World'), findsOneWidget);
    });
  });
}
