import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import '../../../main.dart';

/// Thin wrapper around [FlutterLocalNotificationsPlugin].
class NotificationService {
  NotificationService._();

  static final NotificationService instance = NotificationService._();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  /// Must be called once from [main] before [runApp].
  Future<void> init() async {
    const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosInit = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );
    const settings = InitializationSettings(android: androidInit, iOS: iosInit);

    await _plugin.initialize(
      settings,
      onDidReceiveNotificationResponse: _onTap,
    );

    // Request notification permission on Android 13+
    if (Platform.isAndroid) {
      await _plugin
          .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin
          >()
          ?.requestNotificationsPermission();
    }
  }

  /// Show a notification when verification finishes.
  Future<void> showVerificationComplete({
    required int verificationId,
    required bool success,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'watheq_verification',
      'نتائج التحقق',
      channelDescription: 'إشعارات إكمال عمليات التحقق',
      importance: Importance.high,
      priority: Priority.high,
    );
    const iosDetails = DarwinNotificationDetails();
    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    final title = success ? 'تم التحقق بنجاح ✓' : 'فشل التحقق ✗';
    final body = success
        ? 'اكتملت عملية التحقق #$verificationId بنجاح. اضغط لعرض التفاصيل.'
        : 'فشلت عملية التحقق #$verificationId. اضغط لعرض التفاصيل.';

    await _plugin.show(
      verificationId, // unique notification id
      title,
      body,
      details,
      payload: verificationId.toString(),
    );
  }

  /// Called when user taps a notification.
  static void _onTap(NotificationResponse response) {
    final payload = response.payload;
    if (payload == null || payload.isEmpty) return;
    final verificationId = int.tryParse(payload);
    if (verificationId == null) return;

    // Navigate to verification details screen via global navigator key.
    final nav = MyApp.navigatorKey.currentState;
    if (nav == null) return;
    nav.pushNamed('/verification-details', arguments: verificationId);
  }
}
